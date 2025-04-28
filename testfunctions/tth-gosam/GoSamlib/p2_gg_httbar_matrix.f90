module     p2_gg_httbar_matrix
   use p2_gg_httbar_util, only: square
   use p2_gg_httbar_util_qp, only: square_qp => square
   use p2_gg_httbar_config, only: ki, ki_qp, &
     & include_helicity_avg_factor, include_color_avg_factor, &
     & debug_lo_diagrams, debug_nlo_diagrams, &
     & include_symmetry_factor, &
     & PSP_check, PSP_verbosity, PSP_rescue, PSP_chk_th1, &
     & PSP_chk_th2, PSP_chk_th3, PSP_chk_kfactor, reduction_interoperation, &
     & PSP_chk_li1, PSP_chk_li2, PSP_chk_li3, PSP_chk_li4, &
     & reduction_interoperation_rescue, convert_to_cdr
   use p2_gg_httbar_kinematics, only: &
       in_helicities, symmetry_factor, num_legs, &
       lo_qcd_couplings, corrections_are_qcd, num_light_quarks, num_gluons
   use p2_gg_httbar_model, only: Nf, NC, sqrt2, init_functions
   use p2_gg_httbar_model_qp, only: Nf_qp => Nf, NC_qp => NC, sqrt2_qp => sqrt2, &
     & init_functions_qp => init_functions
   use p2_gg_httbar_color, only: TR, CA, CF, numcs, &
     & incolors, init_color
   use p2_gg_httbar_color_qp, only: TR_qp => TR, CA_qp => CA, CF_qp => CF, &
     & incolors_qp => incolors, init_color_qp => init_color
   use p2_gg_httbar_diagramsh0l0, only: amplitude0l0 => amplitude
   use p2_gg_httbar_diagramsh0l0_qp, only: amplitude0l0_qp => amplitude
   use p2_gg_httbar_amplitudeh0, only: samplitudeh0l1 => samplitude, &
     &   finite_renormalisation0 => finite_renormalisation
   use p2_gg_httbar_amplitudeh0_qp, only: samplitudeh0l1_qp => samplitude, &
     &   finite_renormalisation0_qp => finite_renormalisation
   use p2_gg_httbar_diagramsh4l0, only: amplitude4l0 => amplitude
   use p2_gg_httbar_diagramsh4l0_qp, only: amplitude4l0_qp => amplitude
   use p2_gg_httbar_amplitudeh4, only: samplitudeh4l1 => samplitude, &
     &   finite_renormalisation4 => finite_renormalisation
   use p2_gg_httbar_amplitudeh4_qp, only: samplitudeh4l1_qp => samplitude, &
     &   finite_renormalisation4_qp => finite_renormalisation
   use p2_gg_httbar_diagramsh8l0, only: amplitude8l0 => amplitude
   use p2_gg_httbar_diagramsh8l0_qp, only: amplitude8l0_qp => amplitude
   use p2_gg_httbar_amplitudeh8, only: samplitudeh8l1 => samplitude, &
     &   finite_renormalisation8 => finite_renormalisation
   use p2_gg_httbar_amplitudeh8_qp, only: samplitudeh8l1_qp => samplitude, &
     &   finite_renormalisation8_qp => finite_renormalisation
   use p2_gg_httbar_diagramsh12l0, only: amplitude12l0 => amplitude
   use p2_gg_httbar_diagramsh12l0_qp, only: amplitude12l0_qp => amplitude
   use p2_gg_httbar_amplitudeh12, only: samplitudeh12l1 => samplitude, &
     &   finite_renormalisation12 => finite_renormalisation
   use p2_gg_httbar_amplitudeh12_qp, only: samplitudeh12l1_qp => samplitude, &
     &   finite_renormalisation12_qp => finite_renormalisation
   use p2_gg_httbar_dipoles, only: insertion_operator, insertion_operator_qed
   use p2_gg_httbar_dipoles_qp, only: insertion_operator_qp => insertion_operator, &
     & insertion_operator_qed_qp => insertion_operator_qed

   implicit none
   save

   private

   integer :: banner_ch = 6

   public :: initgolem, exitgolem, samplitude
   public :: samplitudel0, samplitudel1
   public :: ir_subtraction, color_correlated_lo2, spin_correlated_lo2
   public :: OLP_color_correlated, OLP_spin_correlated_lo2




contains
   !---#[ subroutine banner:
   subroutine     banner()
      implicit none

      character(len=74) :: frame = "+" // repeat("-", 72) // "+"

      if (banner_ch .le. 0) return

      write(banner_ch,'(A74)') frame
      write(banner_ch,'(A74)') "|   __   __   ___   __   __  __                   GoSam                  |"
      write(banner_ch,'(A74)') "|  / _) /  \ / __) (  ) (  \/  )          An Automated One-Loop          |"
      write(banner_ch,'(A74)') "| ( (/\( () )\__ \ /__\  )    (          Matrix Element Generator        |"
      write(banner_ch,'(A74)') "|  \__/ \__/ (___/(_)(_)(_/\/\_)        Version 2.1.1 Rev: 33a38ce       |"
      write(banner_ch,'(A74)') "|                                                                        |"
      write(banner_ch,'(A74)') "|                                 (c) The GoSam Collaboration 2011-2016  |"
      write(banner_ch,'(A74)') "|                                                                        |"
      write(banner_ch,'(A74)') "|            AUTHORS:                                                    |"
      write(banner_ch,'(A74)') "|            * Gudrun Heinrich      <gudrun@mpp.mpg.de>                  |"
      write(banner_ch,'(A74)') "|            * Stephen Jones        <s.jones@cern.ch>                    |"
      write(banner_ch,'(A74)') "|            * Matthias Kerner      <mkerner@physik.uzh.ch>              |"
      write(banner_ch,'(A74)') "|            * Vitaly Magerya       <vitaly.magerya@tx97.net>            |"
      write(banner_ch,'(A74)') "|            * Pierpaolo Mastrolia  <Pierpaolo.Mastrolia@cern.ch>        |"
      write(banner_ch,'(A74)') "|            * Giovanni Ossola      <gossola@citytech.cuny.edu>          |"
      write(banner_ch,'(A74)') "|            * Tiziano Peraro       <peraro@mpp.mpg.de>                  |"
      write(banner_ch,'(A74)') "|            * Johannes Schlenk     <johannes.schlenk@psi.ch>            |"
      write(banner_ch,'(A74)') "|            * Francesco Tramontano <Francesco.Tramontano@cern.ch>       |"
      write(banner_ch,'(A74)') "|                                                                        |"
      write(banner_ch,'(A74)') "|            FORMER AUTHORS:                                             |"
      write(banner_ch,'(A74)') "|            * Gavin Cullen         <gavin.cullen@desy.de>               |"
      write(banner_ch,'(A74)') "|            * Hans van Deurzen     <hdeurzen@mpp.mpg.de>                |"
      write(banner_ch,'(A74)') "|            * Nicolas Greiner      <greiner@mpp.mpg.de>                 |"
      write(banner_ch,'(A74)') "|            * Stephan Jahn         <sjahn@mpp.mpg.de>                   |"
      write(banner_ch,'(A74)') "|            * Gionata Luisoni      <luisonig@mpp.mpg.de>                |"
      write(banner_ch,'(A74)') "|            * Edoardo Mirabella    <mirabell@mpp.mpg.de>                |"
      write(banner_ch,'(A74)') "|            * Joscha Reichel       <joscha@mpp.mpg.de>                  |"
      write(banner_ch,'(A74)') "|            * Thomas Reiter        <reiterth@mpp.mpg.de>                |"
      write(banner_ch,'(A74)') "|            * Johann Felix von Soden-Fraunhofen <jfsoden@mpp.mpg.de>    |"
      write(banner_ch,'(A74)') "|                                                                        |"
      write(banner_ch,'(A74)') "|  This program is free software: you can redistribute it and/or modify  |"
      write(banner_ch,'(A74)') "|  it under the terms of the GNU General Public License either           |"
      write(banner_ch,'(A74)') "|  version 3, or (at your option) any later version.                     |"
      write(banner_ch,'(A74)') "|                                                                        |"
      write(banner_ch,'(A74)') "|  Scientific publications prepared using the present version of         |"
      write(banner_ch,'(A74)') "|  GoSam or any modified version of it or any code linking to GoSam      |"
      write(banner_ch,'(A74)') "|  or parts of it should make a clear reference to the publication:      |"
      write(banner_ch,'(A74)') "|                                                                        |"
      write(banner_ch,'(A74)') "|      G. Cullen et al.,                                                 |"
      write(banner_ch,'(A74)') "|      ``GoSam-2.0: a tool for automated one-loop calculations           |"
      write(banner_ch,'(A74)') "|                        within the Standard Model and Beyond'',         |"
      write(banner_ch,'(A74)') "|      Eur. Phys. J. C 74 (2014) 8,  3001                                |"
      write(banner_ch,'(A74)') "|      [arXiv:1404.7096 [hep-ph]].                                       |"
      write(banner_ch,'(A74)') frame

      banner_ch = 0
   end subroutine banner
   !---#] subroutine banner:

   !---#[ subroutine initgolem :
   subroutine     initgolem(is_first,stage,rndseed)
      implicit none
      logical, optional, intent(in) :: is_first
      integer, optional, intent(in) :: stage
      integer, optional, intent(in) :: rndseed
      logical :: init_third_party
      logical :: file_exists, dir_exists
      integer i, j
      character(len=50) :: file_name
      character(len=9)  :: dir_name = "BadPoints"
      character(len=6)  :: file_numb
      character(len=9)  :: file_pre = "gs_badpts"
      character(len=3)  :: file_ext = "log"
      character(len=1)  :: cstage
      character(len=4)  :: crndseed
      i = 1
      file_exists =.true.

      if(present(is_first)) then
         init_third_party = is_first
      else
         init_third_party = .true.
      end if

      if(.not. corrections_are_qcd) then
         PSP_check = .false.
      end if
      if (init_third_party) then
      ! call our banner
      call banner()
      if(PSP_check.and.PSP_rescue.and.PSP_verbosity) then
         inquire(file=dir_name, exist=dir_exists)
         if(.not. dir_exists) then
            call system('mkdir BadPoints')
         end if
         if(present(stage)) then
            write(cstage,'(i1)') stage
            write(crndseed,'(i4)') rndseed
            do j=1,4
               if(crndseed(j:j).eq.' ') crndseed(j:j)='0'
            enddo
            file_name = dir_name//"/"//file_pre//"-"//cstage//"-"//crndseed//"."//file_ext
            open(unit=42, file=file_name, status='replace', action='write')
            write(42,'(A22)') "<?xml version='1.0' ?>"
            write(42,'(A5)')  "<run>"
         else
            do while(file_exists)
               write(file_numb, '(I6.1)') i
               file_name = dir_name//"/"//file_pre//trim(adjustl(file_numb))//"."//file_ext
               inquire(file=file_name, exist=file_exists)
               if(file_exists) then
                  write(*,*) "File ", file_name, " already exists!"
                  i = i+1
               else
                  write(*,*) "Bad points stored in file: ", file_name
                  open(unit=42, file=file_name, status='unknown', action='write')
                  write(42,'(A22)') "<?xml version='1.0' ?>"
                  write(42,'(A5)')  "<run>"
               end if
            enddo
         end if
      end if
      end if

      call init_functions()
      call init_color()

      call init_functions_qp()
      call init_color_qp()

   end subroutine initgolem
   !---#] subroutine initgolem :
   !---#[ subroutine exitgolem :
   subroutine     exitgolem(is_last)
      implicit none
      logical, optional, intent(in) :: is_last

      logical :: exit_third_party

      if(present(is_last)) then
         exit_third_party = is_last
      else
         exit_third_party = .true.
      end if
      if (exit_third_party) then
         if(PSP_check.and.PSP_rescue.and.PSP_verbosity) then
            write(42,'(A6)')  "</run>"
            close(unit=42)
         endif
      end if
   end subroutine exitgolem
   !---#] subroutine exitgolem :

   !---#[ subroutine samplitude :
   subroutine     samplitude(vecs, scale2, amp, prec, fpprec, spprec, numprec, ok, h)
      use p2_gg_httbar_kinematics, only: adjust_kinematics
      use p2_gg_httbar_kinematics_qp, only: adjust_kinematics_qp => adjust_kinematics
      use p2_gg_httbar_model
      implicit none
      real(ki), dimension(5, 4), intent(in) :: vecs
      real(ki_qp), dimension(5, 4) :: vecs_qp, vecsrot_qp
      real(ki), dimension(5, 4) :: vecsrot
      real(ki), intent(in) :: scale2
      real(ki), dimension(1:5), intent(out) :: amp
      real(ki_qp), dimension(1:5) :: amp_qp
      real(ki_qp), dimension(2:3) :: irp_qp
      real(ki_qp) :: scale2_qp, rat2_qp
      real(ki), dimension(0:3) :: scales2
      real(ki), dimension(1:5) :: ampdef, amprot, ampres, ampresrot
      real(ki) :: rat2, kfac, zero, angle
      real(ki), dimension(2:3) :: irp
      integer, intent(out) :: prec 
      integer, intent(out), optional :: fpprec, spprec, numprec
      logical, intent(out), optional :: ok
      integer, intent(in), optional :: h
      integer spprec1, fpprec1, spprec2, fpprec2
      integer tmp_red_int, icheck, i, irot
      ampdef=0.0_ki
      amprot=0.0_ki
      ampres=0.0_ki
      ampresrot=0.0_ki
      icheck = 1
      angle = 1.234_ki
      spprec1 = 18
      spprec2 = 18
      fpprec1 = 18
      fpprec2 = 18
      scales2(:) = (/0.0_ki, &
     &              mH, &
     &              mT, &
     &              mT/)
      tmp_red_int = reduction_interoperation
      if(reduction_interoperation.eq.4) then
         PSP_check=.true.
         PSP_rescue=.true.
         icheck = 3
      else
         numprec = ki
         call samplitudel01(vecs, scale2, ampdef, rat2, ok, h)
         amp = ampdef
      endif
      ! RESCUE SYSTEM
      if(PSP_check) then
         if(icheck.eq.1) then
         call ir_subtraction(vecs, scale2, irp, h)
         if((ampdef(3)-irp(2)) .ne. 0.0_ki) then
            spprec1 = -int(log10(abs((ampdef(3)-irp(2))/irp(2))))
         else
            spprec1 = 16
         endif
         if(ampdef(1) .ne. 0.0_ki) then
            kfac = abs(ampdef(2)/ampdef(1))
         else
            kfac = 0.0_ki
         endif
         ! if(spprec1.lt.PSP_chk_th1) then                                       ! RESCUE
         !    icheck=3
         ! endif
         do irot = 1,5
            vecsrot(irot,1) = vecs(irot,1)
            vecsrot(irot,2) = vecs(irot,2)*Cos(angle)-vecs(irot,3)*Sin(angle)
            vecsrot(irot,3) = vecs(irot,2)*Sin(angle)+vecs(irot,3)*Cos(angle)
            vecsrot(irot,4) = vecs(irot,4)
         enddo
         call adjust_kinematics(vecsrot)
         call samplitudel01(vecsrot, scale2, ampresrot, rat2, ok, h)
         if((ampresrot(2)-ampres(2)) .ne. 0.0_ki) then
            fpprec1 = -int(log10(abs((ampresrot(2)-ampdef(2))/((ampresrot(2)+ampdef(2))/2.0_ki))))
         else
            fpprec1 = 16
         endif
         prec = min(spprec1,fpprec1)
         spprec=spprec1
         fpprec=fpprec1
         ! icheck=3
         if(prec.lt.PSP_chk_th1) then                                       ! RESCUE
            icheck=3
         endif
         ! FIXME: always rescue to catch an example that the check above does not catch
         icheck=3

	 endif ! end if(icheck.eq.1)
         

         if(icheck.eq.3.and.PSP_rescue) then
            numprec = ki_qp

            icheck = 1
            reduction_interoperation = reduction_interoperation_rescue
            scale2_qp = real(scale2,ki_qp)
            vecs_qp = vecs
            ! call refine_momenta_to_qp(5,vecs,vecs_qp,3+1,scales2)
            call adjust_kinematics_qp(vecs_qp)
            call samplitudel01_qp(vecs_qp, scale2_qp, amp_qp, rat2_qp, ok, h)
            call ir_subtraction_qp(vecs_qp, scale2_qp, irp_qp, h)
            ampres = real(amp_qp,ki)
            irp = real(irp_qp,ki)
            amp=ampres
            if((ampres(3)-irp(2)) .ne. 0.0_ki) then
               spprec2 = -int(log10(abs((ampres(3)-irp(2))/irp(2))))
            else
               spprec2 = 16
            endif
            if(ampres(1) .ne. 0.0_ki) then
               kfac = abs(ampres(2)/ampres(1))
            else
               kfac = 0.0_ki
            endif
            if(spprec2.gt.PSP_chk_th1.and. &
                 (kfac.lt.PSP_chk_kfactor.or.PSP_chk_kfactor.lt.0)) icheck=2      ! ACCEPT
            ! if(spprec2.lt.PSP_chk_th1.and.spprec2.ge.PSP_chk_th2 &
            !      .or.(kfac.gt.PSP_chk_kfactor.and.PSP_chk_kfactor.gt.0)) icheck=2 ! ROTATION
            ! if(spprec2.lt.PSP_chk_th2) then                                       ! DISCARD
            ! ! if(spprec2.lt.PSP_chk_th3 &
            ! !      .or.(kfac.gt.PSP_chk_kfactor.and.PSP_chk_kfactor.gt.0)) then ! DISCARD
            !    icheck=3
            !    fpprec2=-10        ! Set -10 as finite part precision
            ! endif
            if(spprec2.lt.PSP_chk_th1.or. &
            (kfac.gt.PSP_chk_kfactor.and.PSP_chk_kfactor.gt.0)) icheck=2 ! ROTATION
            if(icheck.eq.2) then
               do irot = 1,5
                  ! vecsrot(irot,1) = vecs(irot,1)
                  ! vecsrot(irot,2) = vecs(irot,2)*Cos(angle)-vecs(irot,3)*Sin(angle)
                  ! vecsrot(irot,3) = vecs(irot,2)*Sin(angle)+vecs(irot,3)*Cos(angle)
                  ! vecsrot(irot,4) = vecs(irot,4)
                  vecsrot_qp(irot,1) = vecs_qp(irot,1)
                  vecsrot_qp(irot,2) = vecs_qp(irot,2)*Cos(real(angle,ki_qp))-vecs_qp(irot,3)*Sin(real(angle,ki_qp))
                  vecsrot_qp(irot,3) = vecs_qp(irot,2)*Sin(real(angle,ki_qp))+vecs_qp(irot,3)*Cos(real(angle,ki_qp))
                  vecsrot_qp(irot,4) = vecs_qp(irot,4)
               enddo
               ! ! call adjust_kinematics(vecsrot)
               ! call samplitudel01(vecsrot, scale2, ampresrot, rat2, ok, h)
               vecs_qp = vecsrot_qp
               call adjust_kinematics_qp(vecs_qp)
               call samplitudel01_qp(vecs_qp, scale2_qp, amp_qp, rat2_qp, ok, h)
               ampresrot = real(amp_qp,ki)
               if((ampresrot(2)-ampres(2)) .ne. 0.0_ki) then
                  fpprec2 = -int(log10(abs((ampresrot(2)-ampres(2))/((ampresrot(2)+ampres(2))/2.0_ki))))
               else
                  fpprec2 = 16
               endif
               if(fpprec2.ge.PSP_chk_th3) icheck=1                         ! ACCEPTED
               if(fpprec2.lt.PSP_chk_th3) icheck=3                         ! DISCARD
            endif
            reduction_interoperation = tmp_red_int
            prec = min(spprec2,fpprec2)
            spprec=spprec2
            fpprec=fpprec2
         endif

         if(icheck.eq.3.and.PSP_verbosity) then
            write(42,'(2x,A7)')"<event>"
            write(42,'(4x,A15,A12,A3)') &
                 &  "<process name='","p2_gg_httbar","'/>"
            write(42,'(4x,A21,I2.1,A7,I2.1,A7,I2.1,A3)') &
                 &  "<PSP_thresholds th1='", PSP_chk_th1, &
                 &                "' th2='", PSP_chk_th2, &
                 &                "' th3='", PSP_chk_th3,"'/>"
            write(42,'(4x,A16,D23.16,A3)') &
                 &  "<PSP_kfaktor k='", PSP_chk_kfactor,"'/>"
            write(42,'(4x,A15,I3.1,A6,I3.1,A3)') &
                 &  "<PSP_prec1 sp='", spprec1, "' fp='", fpprec1, "'/>"
            write(42,'(4x,A15,I3.1,A6,I3.1,A3)') &
                 &  "<PSP_prec2 sp='", spprec2, "' fp='", fpprec2, "'/>"
            write(42,'(4x,A10,D23.16,A3)') &
                 &  "<born LO='", ampdef(1), "'/>"
            write(42,'(4x,A10,D23.16,A3)') &
                 &  "<rat2 r2='", rat2, "'/>"
            write(42,'(4x,A15,D23.16,A6,D23.16,A6,D23.16,A3)') &
                 &  "<amp       sp='", ampdef(3)   ,"' ir='", irp(2),"' fp='", ampdef(2)   ,"'/>"
            write(42,'(4x,A15,D23.16,A6,D23.16,A6,D23.16,A3)') &
                 &  "<amprot    sp='", amprot(3)   ,"' ir='", irp(2),"' fp='", amprot(2)   ,"'/>"
            write(42,'(4x,A15,D23.16,A6,D23.16,A6,D23.16,A3)') &
                 &  "<ampres    sp='", ampres(3)   ,"' ir='", irp(2),"' fp='", ampres(2)   ,"'/>"
            write(42,'(4x,A15,D23.16,A6,D23.16,A6,D23.16,A3)') &
                 &  "<ampresrot sp='", ampresrot(3),"' ir='", irp(2),"' fp='", ampresrot(2),"'/>"
            write(42,'(4x,A9)') "<momenta>"
            do i=1,5
               write(42,'(8x,A8,3(D23.16,A6),D23.16,A3)') "<mom e='", vecs(i,1), &
                    &  "' px='", vecs(i,2), &
                    &  "' py='", vecs(i,3), &
                    &  "' pz='", vecs(i,4), "'/>"
            enddo
            write(42,'(4x,A10)')"</momenta>"
            write(42,'(2x,A8)')"</event>"
         endif
      else
         prec = 20 ! If PSP_check is off, precision is set to unrealistic value = 20.
      end if
 end subroutine samplitude
   !---#] subroutine samplitude :

   !---#[ subroutine samplitudel01 :
   subroutine     samplitudel01(vecs, scale2, amp, rat2, ok, h)
      use p2_gg_httbar_config, only: &
         & debug_lo_diagrams, debug_nlo_diagrams, logfile, deltaOS, &
         & renormalisation, renorm_beta, renorm_mqwf, renorm_decoupling, &
         & renorm_logs, renorm_mqse, renorm_yukawa, nlo_prefactors
      use p2_gg_httbar_kinematics, only: &
         & inspect_kinematics, init_event
      use p2_gg_httbar_model
      use p2_gg_httbar_dipoles, only: pi
      implicit none
      real(ki), dimension(5, 4), intent(in) :: vecs
      real(ki), intent(in) :: scale2
      real(ki), dimension(5), intent(out) :: amp
      real(ki), intent(out) :: rat2
      logical, intent(out), optional :: ok
      integer, intent(in), optional :: h
      real(ki) :: nlo_coupling

      complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)

      ! Number of heavy quark flavours in loops.
      real(ki), parameter :: NFh = 1.0_ki

      logical :: my_ok

      ! used for m=0 QCD renormalisation
      real(ki) :: beta0

      if(corrections_are_qcd) then
         nlo_coupling = gs*gs
      else
         nlo_coupling = e*e
      end if

      if(debug_lo_diagrams .or. debug_nlo_diagrams) then
         call init_event(vecs)
         write(logfile,'(A7)') "<event>"
         call inspect_kinematics(logfile)
      end if


      if (present(h)) then
         amp((/1,5/)) = samplitudel0(vecs, h)
      else
         amp((/1,5/))   = samplitudel0(vecs)
      end if
      select case (renormalisation)
      case (0)
         ! no renormalisation
         deltaOS = 0.0_ki
      case (1)
         ! fully renormalized
         if(renorm_mqse) then
            deltaOS = 1.0_ki
         else
            deltaOS = 0.0_ki
         end if
      case (2)
         ! massive quark counterterms only
         deltaOS = 1.0_ki
      case default
         ! not implemented
         print*, "In p2_gg_httbar_matrix:"
         print*, "  invalid value for renormalisation=", renormalisation
         stop
      end select

      if (present(h)) then
         amp((/4,3,2/)) = samplitudel1(vecs, scale2, my_ok, rat2, h)/nlo_coupling
      else
         amp((/4,3,2/)) = samplitudel1(vecs, scale2, my_ok, rat2)/nlo_coupling
      end if
      select case (renormalisation)
      case (0)
         ! no renormalisation
      case (1)
         ! fully renormalized
         if(corrections_are_qcd) then
            if (renorm_beta) then
               beta0 = (11.0_ki * CA - 4.0_ki * TR * (NF + NFh)) / 6.0_ki
               amp(3) = amp(3) - lo_qcd_couplings * beta0 * amp(1)
               if (renorm_logs) then
                  amp(2) = amp(2) + lo_qcd_couplings * 4.0_ki * TR / 6.0_ki * &
                      &            log(scale2/mT**2) * amp(1)  
               end if
               amp(2) = amp(2) + lo_qcd_couplings * CA / 6.0_ki * amp(1)
            end if
            if (renorm_yukawa) then
            ! Renormalization of Yukawa coupling
               if (mT > 0.0_ki) then  
                   amp(3) = amp(3) -1.5_ki * CF * amp(1)
                   amp(2) = amp(2) -2.5_ki * CF * amp(1)
                   if (renorm_logs) then
                      amp(2) = amp(2) &
                     &   - (1.5_ki*log(scale2/mT/mT)) * CF * amp(1)
                   end if
                   amp(3) = amp(3) -1.5_ki * CF * amp(1)
                   amp(2) = amp(2) -2.5_ki * CF * amp(1)
                   if (renorm_logs) then
                      amp(2) = amp(2) &
                     &   - (1.5_ki*log(scale2/mT/mT)) * CF * amp(1)
                   end if
               end if
            end if
            if (renorm_mqwf) then
            ! wave function renormalisation:
               amp(3) = amp(3) - 1.5_ki * CF * amp(1)
               amp(2) = amp(2) - 2.5_ki * CF * amp(1)
               if (renorm_logs) then
                  amp(2) = amp(2) &
                 &   - (1.5_ki*log(scale2/mT/mT)) * CF * amp(1)
               end if
               amp(3) = amp(3) - 1.5_ki * CF * amp(1)
               amp(2) = amp(2) - 2.5_ki * CF * amp(1)
               if (renorm_logs) then
                  amp(2) = amp(2) &
                 &   - (1.5_ki*log(scale2/mT/mT)) * CF * amp(1)
               end if
            end if
            if (renorm_decoupling) then
               amp(3) = amp(3) - num_gluons * 2.0_ki * TR / 3.0_ki * NFh * &
                                &  amp(1)

               if (renorm_logs) then
                  amp(2) = amp(2) - num_gluons * 2.0_ki * TR / 3.0_ki * &
                      &            log(scale2/mT**2) * amp(1)  
               end if
            end if
         end if
      case (2)
         ! massive quark counterterms only
      case default
         ! not implemented
         print*, "In p2_gg_httbar_matrix:"
         print*, "  invalid value for renormalisation=", renormalisation
         stop
      end select
      if (convert_to_cdr) then
         ! Scheme conversion for infrared structure
         ! Reference:
         ! S. Catani, M. H. Seymour, Z. Trocsanyi,
         ! ``Regularisation scheme independence and unitarity
         !   in QCD cross-sections,''
         ! Phys.Rev. D 55 (1997) 6819
         ! arXiv:hep-ph/9610553
         amp(2) = amp(2) - amp(1) * (&
           &          num_light_quarks * 0.5_ki * CF &
           &        + num_gluons * 1.0_ki/6.0_ki * CA)
      end if
      if (present(ok)) ok = my_ok

      if(debug_lo_diagrams .or. debug_nlo_diagrams) then
         write(logfile,'(A25,E24.16,A3)') &
            & "<result kind='lo' value='", amp(1), "'/>"
         write(logfile,'(A33,E24.16,A3)') &
            & "<result kind='nlo-finite' value='", amp(2), "'/>"
         write(logfile,'(A33,E24.16,A3)') &
            & "<result kind='nlo-single' value='", amp(3), "'/>"
         write(logfile,'(A33,E24.16,A3)') &
            & "<result kind='nlo-double' value='", amp(4), "'/>"
         if(my_ok) then
            write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
         else
            write(logfile,'(A29)') "<flag name='ok' status='no'/>"
         end if
         write(logfile,'(A8)') "</event>"
      end if
      select case(nlo_prefactors)
      case(0)
         ! The result is already in its desired form
      case(1)
         amp(2:4) = amp(2:4) * nlo_coupling
      case(2)
         amp(2:4) = amp(2:4) * nlo_coupling / 8.0_ki / pi / pi
      end select
   end subroutine samplitudel01
   !---#] subroutine samplitudel01 :
   !---#[ function samplitudel0 :
   function     samplitudel0(vecs, h) result(amp)
      use p2_gg_httbar_config, only: logfile
      use p2_gg_httbar_kinematics, only: init_event
      use p2_gg_httbar_color, only: T4T5
      implicit none
      real(ki), dimension(5, 4), intent(in) :: vecs
      integer, optional, intent(in) :: h
      real(ki), dimension(2) :: amp
      real(ki) heli_amp
      complex(ki), dimension(numcs) :: color_vector
      logical, dimension(0:15) :: eval_heli
      real(ki), dimension(5, 4) :: pvecs

      if (present(h)) then
         eval_heli(:) = .false.
         eval_heli(h) = .true.
      else
         eval_heli(:) = .true.
      end if

      amp = 0.0_ki
      if (eval_heli(0)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='0' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(1)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='1' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(2)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='2' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(3)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='3' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(4)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='4' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(5)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='5' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(6)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='6' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(7)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='7' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(8)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='8' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(9)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='9' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(10)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='10' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(11)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='11' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(12)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='12' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(13)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='13' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(14)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='14' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (eval_heli(15)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='15' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0()
         heli_amp = square(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square(color_vector, T4T5)
      end if
      if (include_helicity_avg_factor) then
         amp = amp / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         amp = amp / incolors
      end if
      if (include_symmetry_factor) then
         amp = amp / real(symmetry_factor, ki)
      end if
   end function samplitudel0
   !---#] function samplitudel0 :
   !---#[ function samplitudel1 :
   function     samplitudel1(vecs,scale2,ok,rat2,h) result(amp)
      use p2_gg_httbar_config, only: &
         & debug_nlo_diagrams, logfile, renorm_gamma5
      use p2_gg_httbar_kinematics, only: init_event
      implicit none
      real(ki), dimension(5, 4), intent(in) :: vecs
      logical, intent(out) :: ok
      real(ki), intent(in) :: scale2
      real(ki), intent(out) :: rat2
      integer, optional, intent(in) :: h
      real(ki), dimension(5, 4) :: pvecs
      real(ki), dimension(-2:0) :: amp, heli_amp
      logical :: my_ok
      logical, dimension(0:15) :: eval_heli
      real(ki) :: fr, rational2

      amp(:) = 0.0_ki
      rat2 = 0.0_ki
      ok = .true.

      if (present(h)) then
         eval_heli(:) = .false.
         eval_heli(h) = .true.
      else
         eval_heli(:) = .true.
      end if
      if (eval_heli(0)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='0'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(1)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='1'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(2)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='2'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(3)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='3'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(4)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='4'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(5)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='5'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(6)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='6'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(7)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='7'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(8)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='8'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(9)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='9'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(10)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='10'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(11)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='11'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(12)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='12'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(13)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='13'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(14)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='14'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(15)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='15'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1(real(scale2,ki),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12(real(scale2,ki))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (include_helicity_avg_factor) then
         amp = amp / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         amp = amp / incolors
      end if
      if (include_symmetry_factor) then
         amp = amp / real(symmetry_factor, ki)
      end if
   end function samplitudel1
   !---#] function samplitudel1 :
   !---#[ subroutine ir_subtraction :
   subroutine     ir_subtraction(vecs,scale2,amp,h)
      use p2_gg_httbar_config, only: &
         & nlo_prefactors
      use p2_gg_httbar_dipoles, only: pi
      use p2_gg_httbar_kinematics, only: &
         & init_event, corrections_are_qcd
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: T4T5
      implicit none
      real(ki), dimension(5, 4), intent(in) :: vecs
      real(ki), intent(in) :: scale2
      integer, optional, intent(in) :: h
      real(ki), dimension(2), intent(out) :: amp
      real(ki), dimension(2) :: heli_amp
      real(ki), dimension(5, 4) :: pvecs
      complex(ki), dimension(numcs,numcs,2) :: oper
      complex(ki), dimension(numcs) :: color_vectorl0, pcolor
      logical, dimension(0:15) :: eval_heli
      real(ki) :: nlo_coupling

      if (present(h)) then
         eval_heli(:) = .false.
         eval_heli(h) = .true.
      else
         eval_heli(:) = .true.
      end if

      if(corrections_are_qcd) then
         nlo_coupling = gs*gs
      else
         nlo_coupling = e*e
      end if

      if (corrections_are_qcd) then
        oper = insertion_operator(real(scale2,ki), vecs)
      else
        oper = insertion_operator_qed(real(scale2,ki), vecs)
      endif
      amp(:) = 0.0_ki
      if (eval_heli(0)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(1)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(2)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(3)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(4)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(5)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(6)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(7)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(8)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(9)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(10)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(11)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(12)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(13)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(14)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(15)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (include_helicity_avg_factor) then
         amp = amp / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         amp = amp / incolors
      end if
      if (include_symmetry_factor) then
         amp = amp / real(symmetry_factor, ki)
      end if
      select case(nlo_prefactors)
      case(0)
         ! The result is already in its desired form
      case(1)
         amp(:) = amp(:) * nlo_coupling
      case(2)
         amp(:) = amp(:) * nlo_coupling / 8.0_ki / pi / pi
      end select
   end subroutine ir_subtraction
   !---#] subroutine ir_subtraction :
   !---#[ subroutine samplitudel01_qp :
   subroutine     samplitudel01_qp(vecs, scale2, amp, rat2, ok, h)
      use p2_gg_httbar_config, only: &
         & debug_lo_diagrams, debug_nlo_diagrams, logfile, deltaOS, &
         & renormalisation, renorm_beta, renorm_mqwf, renorm_decoupling, &
         & renorm_logs, renorm_mqse, renorm_yukawa, nlo_prefactors
      use p2_gg_httbar_kinematics_qp, only: &
         & inspect_kinematics, init_event
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_dipoles_qp, only: pi
      implicit none
      real(ki_qp), dimension(5, 4), intent(in) :: vecs
      real(ki_qp), intent(in) :: scale2
      real(ki_qp), dimension(5), intent(out) :: amp
      real(ki_qp), intent(out) :: rat2
      logical, intent(out), optional :: ok
      integer, intent(in), optional :: h
      real(ki_qp) :: nlo_coupling

      complex(ki_qp), parameter :: i_ = (0.0_ki_qp, 1.0_ki_qp)

      ! Number of heavy quark flavours in loops.
      real(ki_qp), parameter :: NFh_qp = 1.0_ki_qp

      logical :: my_ok

      ! used for m=0 QCD renormalisation
      real(ki_qp) :: beta0

      if(corrections_are_qcd) then
         nlo_coupling = gs*gs
      else
         nlo_coupling = e*e
      end if

      if(debug_lo_diagrams .or. debug_nlo_diagrams) then
         call init_event(vecs)
         write(logfile,'(A7)') "<event>"
         call inspect_kinematics(logfile)
      end if

      
      if (present(h)) then
         amp((/1,5/)) = samplitudel0_qp(vecs, h)
      else
         amp((/1,5/))   = samplitudel0_qp(vecs)
      end if
      select case (renormalisation)
      case (0)
         ! no renormalisation
         deltaOS = 0.0_ki_qp
      case (1)
         ! fully renormalized
         if(renorm_mqse) then
            deltaOS = 1.0_ki_qp
         else
            deltaOS = 0.0_ki_qp
         end if
      case (2)
         ! massive quark counterterms only
         deltaOS = 1.0_ki_qp
      case default
         ! not implemented
         print*, "In p2_gg_httbar_matrix:"
         print*, "  invalid value for renormalisation=", renormalisation
         stop
      end select

      if (present(h)) then
         amp((/4,3,2/)) = samplitudel1_qp(vecs, scale2, my_ok, rat2, h)/nlo_coupling
      else
         amp((/4,3,2/)) = samplitudel1_qp(vecs, scale2, my_ok, rat2)/nlo_coupling
      end if
      select case (renormalisation)
      case (0)
         ! no renormalisation
      case (1)
         ! fully renormalized
         if(corrections_are_qcd) then
            if (renorm_beta) then
               beta0 = (11.0_ki_qp * CA_qp - 4.0_ki_qp * TR_qp * (NF_qp + NFh_qp)) / 6.0_ki_qp
               amp(3) = amp(3) - lo_qcd_couplings * beta0 * amp(1)
               if (renorm_logs) then
                  amp(2) = amp(2) + lo_qcd_couplings * 4.0_ki_qp * TR_qp / 6.0_ki_qp * &
                      &            log(scale2/mT**2) * amp(1)  
               end if
               amp(2) = amp(2) + lo_qcd_couplings * CA_qp / 6.0_ki_qp * amp(1)
            end if
            if (renorm_yukawa) then
            ! Renormalization of Yukawa coupling
               if (mT > 0.0_ki_qp) then  
                  amp(3) = amp(3) -1.5_ki_qp * CF_qp * amp(1)
                  amp(2) = amp(2) -2.5_ki_qp * CF_qp * amp(1)
                  if (renorm_logs) then
                     amp(2) = amp(2) &
                    &   - (1.5_ki_qp*log(scale2/mT/mT)) * CF_qp * amp(1)
                  end if
                  amp(3) = amp(3) -1.5_ki_qp * CF_qp * amp(1)
                  amp(2) = amp(2) -2.5_ki_qp * CF_qp * amp(1)
                  if (renorm_logs) then
                     amp(2) = amp(2) &
                    &   - (1.5_ki_qp*log(scale2/mT/mT)) * CF_qp * amp(1)
                  end if
               end if
            end if
            if (renorm_mqwf) then
            ! wave function renormalisation:
               amp(3) = amp(3) - 1.5_ki_qp * CF_qp * amp(1)
               amp(2) = amp(2) - 2.5_ki_qp * CF_qp * amp(1)
               if (renorm_logs) then
                  amp(2) = amp(2) &
                 &   - (1.5_ki_qp*log(scale2/mT/mT)) * CF_qp * amp(1)
               end if
               amp(3) = amp(3) - 1.5_ki_qp * CF_qp * amp(1)
               amp(2) = amp(2) - 2.5_ki_qp * CF_qp * amp(1)
               if (renorm_logs) then
                  amp(2) = amp(2) &
                 &   - (1.5_ki_qp*log(scale2/mT/mT)) * CF_qp * amp(1)
               end if
            end if
            if (renorm_decoupling) then
               amp(3) = amp(3) - num_gluons * 2.0_ki_qp * TR_qp / 3.0_ki_qp * NFh_qp * &
                                &  amp(1)

               if (renorm_logs) then
                  amp(2) = amp(2) - num_gluons * 2.0_ki_qp * TR_qp / 3.0_ki_qp * &
                      &            log(scale2/mT**2) * amp(1)  
               end if
            end if
         end if
      case (2)
         ! massive quark counterterms only
      case default
         ! not implemented
         print*, "In p2_gg_httbar_matrix:"
         print*, "  invalid value for renormalisation=", renormalisation
         stop
      end select
      if (convert_to_cdr) then
         ! Scheme conversion for infrared structure
         ! Reference:
         ! S. Catani, M. H. Seymour, Z. Trocsanyi,
         ! ``Regularisation scheme independence and unitarity
         !   in QCD cross-sections,''
         ! Phys.Rev. D 55 (1997) 6819
         ! arXiv:hep-ph/9610553
         amp(2) = amp(2) - amp(1) * (&
           &          num_light_quarks * 0.5_ki_qp * CF_qp &
           &        + num_gluons * 1.0_ki_qp/6.0_ki_qp * CA_qp)
      end if
      if (present(ok)) ok = my_ok

      if(debug_lo_diagrams .or. debug_nlo_diagrams) then
         write(logfile,'(A25,E24.16,A3)') &
            & "<result kind='lo' value='", amp(1), "'/>"
         write(logfile,'(A33,E24.16,A3)') &
            & "<result kind='nlo-finite' value='", amp(2), "'/>"
         write(logfile,'(A33,E24.16,A3)') &
            & "<result kind='nlo-single' value='", amp(3), "'/>"
         write(logfile,'(A33,E24.16,A3)') &
            & "<result kind='nlo-double' value='", amp(4), "'/>"
         if(my_ok) then
            write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
         else
            write(logfile,'(A29)') "<flag name='ok' status='no'/>"
         end if
         write(logfile,'(A8)') "</event>"
      end if
      select case(nlo_prefactors)
      case(0)
         ! The result is already in its desired form
      case(1)
         amp(2:4) = amp(2:4) * nlo_coupling
      case(2)
         amp(2:4) = amp(2:4) * nlo_coupling / 8.0_ki_qp / pi / pi
      end select
   end subroutine samplitudel01_qp
   !---#] subroutine samplitudel01_qp :
   !---#[ function samplitudel0_qp :
   function     samplitudel0_qp(vecs, h) result(amp)
      use p2_gg_httbar_config, only: logfile
      use p2_gg_httbar_kinematics_qp, only: init_event
      use p2_gg_httbar_color_qp, only: T4T5
      implicit none
      real(ki_qp), dimension(5, 4), intent(in) :: vecs
      integer, optional, intent(in) :: h
      real(ki_qp), dimension(2) :: amp
      real(ki_qp) :: heli_amp
      complex(ki_qp), dimension(numcs) :: color_vector
      logical, dimension(0:15) :: eval_heli
      real(ki_qp), dimension(5, 4) :: pvecs

      if (present(h)) then
         eval_heli(:) = .false.
         eval_heli(h) = .true.
      else
         eval_heli(:) = .true.
      end if

      amp = 0.0_ki_qp
      if (eval_heli(0)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='0' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(1)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='1' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(2)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='2' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(3)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='3' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude0l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(4)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='4' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(5)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='5' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(6)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='6' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(7)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='7' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude4l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(8)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='8' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(9)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='9' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(10)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='10' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(11)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='11' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude8l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(12)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='12' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(13)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='13' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(14)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='14' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (eval_heli(15)) then
         if (debug_lo_diagrams) then
            write(logfile,*) "<helicity index='15' >"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         color_vector = amplitude12l0_qp()
         heli_amp = square_qp(color_vector)
         if (debug_lo_diagrams) then
            write(logfile,'(A25,E24.16,A3)') &
                & "<result kind='lo' value='", heli_amp, "'/>"
            write(logfile,*) "</helicity>"
         end if
         amp(1) = amp(1) + heli_amp
         amp(2) = amp(2) + square_qp(color_vector, T4T5)
      end if
      if (include_helicity_avg_factor) then
         amp = amp / real(in_helicities, ki_qp)
      end if
      if (include_color_avg_factor) then
         amp = amp / incolors
      end if
      if (include_symmetry_factor) then
         amp = amp / real(symmetry_factor, ki_qp)
      end if
   end function samplitudel0_qp
   !---#] function samplitudel0_qp :
   !---#[ function samplitudel1_qp :
   function     samplitudel1_qp(vecs,scale2,ok,rat2,h) result(amp)
      use p2_gg_httbar_config, only: &
         & debug_nlo_diagrams, logfile, renorm_gamma5
      use p2_gg_httbar_kinematics_qp, only: init_event
      implicit none
      real(ki_qp), dimension(5, 4), intent(in) :: vecs
      logical, intent(out) :: ok
      real(ki_qp), intent(in) :: scale2
      real(ki_qp), intent(out) :: rat2
      integer, optional, intent(in) :: h
      real(ki_qp), dimension(5, 4) :: pvecs
      real(ki_qp), dimension(-2:0) :: amp, heli_amp
      logical :: my_ok
      logical, dimension(0:15) :: eval_heli
      real(ki_qp) :: fr, rational2

      amp(:) = 0.0_ki_qp
      rat2 = 0.0_ki_qp
      ok = .true.

      if (present(h)) then
         eval_heli(:) = .false.
         eval_heli(h) = .true.
      else
         eval_heli(:) = .true.
      end if


      if (eval_heli(0)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='0'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(1)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='1'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(2)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='2'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(3)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='3'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh0l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation0_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(4)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='4'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(5)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='5'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(6)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='6'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(7)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='7'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh4l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation4_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(8)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='8'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(9)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='9'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(10)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='10'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(11)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='11'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh8l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation8_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(12)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='12'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(13)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='13'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, -1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(14)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='14'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, -1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (eval_heli(15)) then
         if(debug_nlo_diagrams) then
            write(logfile,*) "<helicity index='15'>"
         end if
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         heli_amp = samplitudeh12l1_qp(real(scale2,ki_qp),my_ok,rational2)
         if (corrections_are_qcd .and. renorm_gamma5) then
            !---#[ reinitialize kinematics:
            pvecs(1,:) = vecs(1,:)
            pvecs(2,:) = vecs(2,:)
            pvecs(3,:) = vecs(3,:)
            pvecs(4,:) = vecs(4,:)
            pvecs(5,:) = vecs(5,:)
            call init_event(pvecs, +1, +1)
            !---#] reinitialize kinematics:
            fr = finite_renormalisation12_qp(real(scale2,ki_qp))
            heli_amp(0) = heli_amp(0) + fr
         end if
         ok = ok .and. my_ok
         amp = amp + heli_amp
         rat2 = rat2 + rational2

         if(debug_nlo_diagrams) then
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-finite' value='", heli_amp(0), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-single' value='", heli_amp(-1), "'/>"
            write(logfile,'(A33,E24.16,A3)') &
                & "<result kind='nlo-double' value='", heli_amp(-2), "'/>"
            if (corrections_are_qcd .and. renorm_gamma5) then
               write(logfile,'(A30,E24.16,A3)') &
                   & "<result kind='fin-ren' value='", fr, "'/>"
            end if
            if(my_ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</helicity>"
         end if
      end if
      if (include_helicity_avg_factor) then
         amp = amp / real(in_helicities, ki_qp)
      end if
      if (include_color_avg_factor) then
         amp = amp / incolors
      end if
      if (include_symmetry_factor) then
         amp = amp / real(symmetry_factor, ki_qp)
      end if
   end function samplitudel1_qp
   !---#] function samplitudel1_qp :
   !---#[ subroutine ir_subtraction_qp :
   subroutine     ir_subtraction_qp(vecs,scale2,amp,h)
      use p2_gg_httbar_config, only: &
         & nlo_prefactors
      use p2_gg_httbar_dipoles_qp, only: pi
      use p2_gg_httbar_kinematics_qp, only: &
         & init_event, corrections_are_qcd
      use p2_gg_httbar_model_qp
      implicit none
      real(ki_qp), dimension(5, 4), intent(in) :: vecs
      real(ki_qp), intent(in) :: scale2
      integer, optional, intent(in) :: h
      real(ki_qp), dimension(2), intent(out) :: amp
      real(ki_qp), dimension(2) :: heli_amp
      real(ki_qp), dimension(5, 4) :: pvecs
      complex(ki_qp), dimension(numcs,numcs,2) :: oper
      complex(ki_qp), dimension(numcs) :: color_vectorl0, pcolor
      logical, dimension(0:15) :: eval_heli
      real(ki_qp) :: nlo_coupling

      if (present(h)) then
         eval_heli(:) = .false.
         eval_heli(h) = .true.
      else
         eval_heli(:) = .true.
      end if

      if(corrections_are_qcd) then
         nlo_coupling = gs*gs
      else
         nlo_coupling = e*e
      end if

      if (corrections_are_qcd) then
        oper = insertion_operator_qp(real(scale2,ki_qp), vecs)
      else
        oper = insertion_operator_qed_qp(real(scale2,ki_qp), vecs)
      endif
      amp(:) = 0.0_ki_qp
      if (eval_heli(0)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(1)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(2)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(3)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude0l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(4)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(5)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(6)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(7)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude4l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(8)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(9)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(10)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(11)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude8l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(12)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(13)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, -1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(14)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, -1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (eval_heli(15)) then
         !---#[ reinitialize kinematics:
         pvecs(1,:) = vecs(1,:)
         pvecs(2,:) = vecs(2,:)
         pvecs(3,:) = vecs(3,:)
         pvecs(4,:) = vecs(4,:)
         pvecs(5,:) = vecs(5,:)
         call init_event(pvecs, +1, +1)
         !---#] reinitialize kinematics:
         pcolor = amplitude12l0_qp()
         color_vectorl0(1) = pcolor(1)
         color_vectorl0(2) = pcolor(2)
         color_vectorl0(3) = pcolor(3)
         if (corrections_are_qcd) then
           heli_amp(1) = square_qp(color_vectorl0, oper(:,:,1))
           heli_amp(2) = square_qp(color_vectorl0, oper(:,:,2))
         else
           heli_amp(1) = square_qp(color_vectorl0)*oper(1,1,1)
           heli_amp(2) = square_qp(color_vectorl0)*oper(1,1,2)
         endif
         amp = amp + heli_amp
      endif
      if (include_helicity_avg_factor) then
         amp = amp / real(in_helicities, ki_qp)
      end if
      if (include_color_avg_factor) then
         amp = amp / incolors
      end if
      if (include_symmetry_factor) then
         amp = amp / real(symmetry_factor, ki_qp)
      end if
      select case(nlo_prefactors)
      case(0)
         ! The result is already in its desired form
      case(1)
         amp(:) = amp(:) * nlo_coupling
      case(2)
         amp(:) = amp(:) * nlo_coupling / 8.0_ki_qp / pi / pi
      end select
   end subroutine ir_subtraction_qp
   !---#] subroutine ir_subtraction_qp :
   !---#[ color correlated ME :
   pure subroutine color_correlated_lo(color_vector,res)
      use p2_gg_httbar_color, only: T1T1, &
      & T1T2, &
      & T1T4, &
      & T1T5, &
      & T2T2, &
      & T2T4, &
      & T2T5, &
      & T4T4, &
      & T4T5, &
      & T5T5
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector
      real(ki), dimension(num_legs,num_legs), intent(out) :: res
      res(:,:)=0.0_ki
      res(1,1) = square(color_vector,T1T1)
      res(1,1) = res(1,1)
      res(1,2) = square(color_vector,T1T2)
      res(2,1) = res(1,2)
      res(1,4) = square(color_vector,T1T4)
      res(4,1) = res(1,4)
      res(1,5) = square(color_vector,T1T5)
      res(5,1) = res(1,5)
      res(2,2) = square(color_vector,T2T2)
      res(2,2) = res(2,2)
      res(2,4) = square(color_vector,T2T4)
      res(4,2) = res(2,4)
      res(2,5) = square(color_vector,T2T5)
      res(5,2) = res(2,5)
      res(4,4) = square(color_vector,T4T4)
      res(4,4) = res(4,4)
      res(4,5) = square(color_vector,T4T5)
      res(5,4) = res(4,5)
      res(5,5) = square(color_vector,T5T5)
      res(5,5) = res(5,5)
   end subroutine color_correlated_lo

   subroutine     color_correlated_lo2(vecs,borncc)
      use p2_gg_httbar_kinematics, only: init_event
      implicit none
      real(ki), dimension(num_legs, 4), intent(in) :: vecs
      real(ki), dimension(num_legs,num_legs), intent(out) :: borncc
      real(ki), dimension(num_legs,num_legs) :: borncc_heli
      real(ki), dimension(num_legs, 4) :: pvecs
      complex(ki), dimension(numcs) :: color_vector

      borncc(:,:) = 0.0_ki
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call color_correlated_lo(color_vector,borncc_heli)
      ! The minus is part in the definition according to PowHEG Box.
      ! Since they use it we include it:
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call color_correlated_lo(color_vector,borncc_heli)
      borncc(:,:) = borncc(:,:) - borncc_heli(:,:)
      if (include_helicity_avg_factor) then
         borncc = borncc / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         borncc = borncc / incolors
      end if
      if (include_symmetry_factor) then
         borncc = borncc / real(symmetry_factor, ki)
      end if
   end subroutine color_correlated_lo2


   pure subroutine OLP_color_correlated_lo(color_vector,res)
      use p2_gg_httbar_color, only: T1T1, &
      & T1T2, &
      & T1T4, &
      & T1T5, &
      & T2T2, &
      & T2T4, &
      & T2T5, &
      & T4T4, &
      & T4T5, &
      & T5T5
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector
      real(ki), dimension(num_legs*(num_legs-1)/2), intent(out) :: res
      res(:)=0.0_ki   
      res(1) = square(color_vector,T1T2)  
      res(4) = square(color_vector,T1T4)  
      res(7) = square(color_vector,T1T5)    
      res(5) = square(color_vector,T2T4)  
      res(8) = square(color_vector,T2T5)    
      res(10) = square(color_vector,T4T5)   
   end subroutine OLP_color_correlated_lo


   subroutine OLP_color_correlated(vecs,ampcc)
      use p2_gg_httbar_kinematics, only: init_event
      implicit none
      real(ki), dimension(num_legs, 4), intent(in) :: vecs
      real(ki), dimension(num_legs*(num_legs-1)/2), intent(out) :: ampcc
      real(ki), dimension(num_legs,num_legs) :: borncc
      real(ki), dimension(num_legs*(num_legs-1)/2) :: ampcc_heli
      real(ki), dimension(num_legs, 4) :: pvecs
      complex(ki), dimension(numcs) :: color_vector
      ampcc(:) = 0.0_ki
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude0l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude4l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude8l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      color_vector = amplitude12l0()
      call OLP_color_correlated_lo(color_vector,ampcc_heli)

      ampcc(:) = ampcc(:) + ampcc_heli(:)
      
      if (include_helicity_avg_factor) then
         ampcc = ampcc / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         ampcc = ampcc / incolors
      end if
      if (include_symmetry_factor) then
         ampcc = ampcc / real(symmetry_factor, ki)
      end if

   end subroutine OLP_color_correlated


   !---#] color correlated ME :
   !---#[ spin correlated ME :
   subroutine spin_correlated_lo2(vecs, bornsc)
      use p2_gg_httbar_kinematics
      implicit none
      real(ki), dimension(num_legs, 4), intent(in) :: vecs
      real(ki), dimension(num_legs,4,4) :: bornsc
      real(ki), dimension(num_legs, 4) :: pvecs
      complex(ki), dimension(4,4) :: tens
      complex(ki) :: pp, pm, mp, mm
      complex(ki), dimension(numcs) :: heli_amp0
      complex(ki), dimension(numcs) :: heli_amp1
      complex(ki), dimension(numcs) :: heli_amp2
      complex(ki), dimension(numcs) :: heli_amp3
      complex(ki), dimension(numcs) :: heli_amp4
      complex(ki), dimension(numcs) :: heli_amp5
      complex(ki), dimension(numcs) :: heli_amp6
      complex(ki), dimension(numcs) :: heli_amp7
      complex(ki), dimension(numcs) :: heli_amp8
      complex(ki), dimension(numcs) :: heli_amp9
      complex(ki), dimension(numcs) :: heli_amp10
      complex(ki), dimension(numcs) :: heli_amp11
      complex(ki), dimension(numcs) :: heli_amp12
      complex(ki), dimension(numcs) :: heli_amp13
      complex(ki), dimension(numcs) :: heli_amp14
      complex(ki), dimension(numcs) :: heli_amp15
      complex(ki), dimension(4) :: eps1
      complex(ki), dimension(4) :: eps2

      bornsc(:,:,:) = 0.0_ki
      !---#[ Initialize helicity amplitudes :
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp0 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp1 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp2 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp3 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp4 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp5 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp6 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp7 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp8 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp9 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp10 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp11 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp12 = amplitude12l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp13 = amplitude12l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp14 = amplitude12l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp15 = amplitude12l0()
      !---#] Initialize helicity amplitudes :
      !---#[ Initialize polarization vectors :
      eps1 = spvak2k1/Spaa(k2,k1)/sqrt2
      eps2 = spvak1k2/Spaa(k1,k2)/sqrt2
      !---#] Initialize polarization vectors :
      ! Note: By omitting the imaginary parts we lose a term:
      !   Imag(B_j(mu,nu)) = i_ * e_(k_j, mu, q_j, nu) * |Born|^2
      ! where q_j is the reference momentum chosen for the paticle
      ! of momentum k_j. This term should, however not be phenomenologically
      ! relevant.
      !---#[ particle 1 :
      pp  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp1,heli_amp1) &
      &          + square_0l_0l_sc(heli_amp3,heli_amp3) &
      &          + square_0l_0l_sc(heli_amp5,heli_amp5) &
      &          + square_0l_0l_sc(heli_amp7,heli_amp7) &
      &          + square_0l_0l_sc(heli_amp9,heli_amp9) &
      &          + square_0l_0l_sc(heli_amp11,heli_amp11) &
      &          + square_0l_0l_sc(heli_amp13,heli_amp13) &
      &          + square_0l_0l_sc(heli_amp15,heli_amp15)
      pm  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp1,heli_amp0) &
      &          + square_0l_0l_sc(heli_amp3,heli_amp2) &
      &          + square_0l_0l_sc(heli_amp5,heli_amp4) &
      &          + square_0l_0l_sc(heli_amp7,heli_amp6) &
      &          + square_0l_0l_sc(heli_amp9,heli_amp8) &
      &          + square_0l_0l_sc(heli_amp11,heli_amp10) &
      &          + square_0l_0l_sc(heli_amp13,heli_amp12) &
      &          + square_0l_0l_sc(heli_amp15,heli_amp14)
      mp  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp0,heli_amp1) &
      &          + square_0l_0l_sc(heli_amp2,heli_amp3) &
      &          + square_0l_0l_sc(heli_amp4,heli_amp5) &
      &          + square_0l_0l_sc(heli_amp6,heli_amp7) &
      &          + square_0l_0l_sc(heli_amp8,heli_amp9) &
      &          + square_0l_0l_sc(heli_amp10,heli_amp11) &
      &          + square_0l_0l_sc(heli_amp12,heli_amp13) &
      &          + square_0l_0l_sc(heli_amp14,heli_amp15)
      mm  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp0,heli_amp0) &
      &          + square_0l_0l_sc(heli_amp2,heli_amp2) &
      &          + square_0l_0l_sc(heli_amp4,heli_amp4) &
      &          + square_0l_0l_sc(heli_amp6,heli_amp6) &
      &          + square_0l_0l_sc(heli_amp8,heli_amp8) &
      &          + square_0l_0l_sc(heli_amp10,heli_amp10) &
      &          + square_0l_0l_sc(heli_amp12,heli_amp12) &
      &          + square_0l_0l_sc(heli_amp14,heli_amp14)

      call construct_polarization_tensor(conjg(eps1),eps1,tens)
      bornsc(1,:,:) = bornsc(1,:,:) + real(tens(:,:) * pp, ki)
      call construct_polarization_tensor(conjg(eps1),conjg(eps1),tens)
      bornsc(1,:,:) = bornsc(1,:,:) + real(tens(:,:) * pm, ki)
      call construct_polarization_tensor(eps1,eps1,tens)
      bornsc(1,:,:) = bornsc(1,:,:) + real(tens(:,:) * mp, ki)
      call construct_polarization_tensor(eps1,conjg(eps1),tens)
      bornsc(1,:,:) = bornsc(1,:,:) + real(tens(:,:) * mm, ki)
      !---#] particle 1 :
      !---#[ particle 2 :
      pp  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp2,heli_amp2) &
      &          + square_0l_0l_sc(heli_amp3,heli_amp3) &
      &          + square_0l_0l_sc(heli_amp6,heli_amp6) &
      &          + square_0l_0l_sc(heli_amp7,heli_amp7) &
      &          + square_0l_0l_sc(heli_amp10,heli_amp10) &
      &          + square_0l_0l_sc(heli_amp11,heli_amp11) &
      &          + square_0l_0l_sc(heli_amp14,heli_amp14) &
      &          + square_0l_0l_sc(heli_amp15,heli_amp15)
      pm  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp2,heli_amp0) &
      &          + square_0l_0l_sc(heli_amp3,heli_amp1) &
      &          + square_0l_0l_sc(heli_amp6,heli_amp4) &
      &          + square_0l_0l_sc(heli_amp7,heli_amp5) &
      &          + square_0l_0l_sc(heli_amp10,heli_amp8) &
      &          + square_0l_0l_sc(heli_amp11,heli_amp9) &
      &          + square_0l_0l_sc(heli_amp14,heli_amp12) &
      &          + square_0l_0l_sc(heli_amp15,heli_amp13)
      mp  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp0,heli_amp2) &
      &          + square_0l_0l_sc(heli_amp1,heli_amp3) &
      &          + square_0l_0l_sc(heli_amp4,heli_amp6) &
      &          + square_0l_0l_sc(heli_amp5,heli_amp7) &
      &          + square_0l_0l_sc(heli_amp8,heli_amp10) &
      &          + square_0l_0l_sc(heli_amp9,heli_amp11) &
      &          + square_0l_0l_sc(heli_amp12,heli_amp14) &
      &          + square_0l_0l_sc(heli_amp13,heli_amp15)
      mm  = 0.0_ki &
      &          + square_0l_0l_sc(heli_amp0,heli_amp0) &
      &          + square_0l_0l_sc(heli_amp1,heli_amp1) &
      &          + square_0l_0l_sc(heli_amp4,heli_amp4) &
      &          + square_0l_0l_sc(heli_amp5,heli_amp5) &
      &          + square_0l_0l_sc(heli_amp8,heli_amp8) &
      &          + square_0l_0l_sc(heli_amp9,heli_amp9) &
      &          + square_0l_0l_sc(heli_amp12,heli_amp12) &
      &          + square_0l_0l_sc(heli_amp13,heli_amp13)

      call construct_polarization_tensor(conjg(eps2),eps2,tens)
      bornsc(2,:,:) = bornsc(2,:,:) + real(tens(:,:) * pp, ki)
      call construct_polarization_tensor(conjg(eps2),conjg(eps2),tens)
      bornsc(2,:,:) = bornsc(2,:,:) + real(tens(:,:) * pm, ki)
      call construct_polarization_tensor(eps2,eps2,tens)
      bornsc(2,:,:) = bornsc(2,:,:) + real(tens(:,:) * mp, ki)
      call construct_polarization_tensor(eps2,conjg(eps2),tens)
      bornsc(2,:,:) = bornsc(2,:,:) + real(tens(:,:) * mm, ki)
      !---#] particle 2 :

      
      if (include_helicity_avg_factor) then
         bornsc = bornsc / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         bornsc = bornsc / incolors
      end if
      if (include_symmetry_factor) then
         bornsc = bornsc / real(symmetry_factor, ki)
      end if
   end subroutine spin_correlated_lo2




   subroutine OLP_spin_correlated_lo2(vecs, ampsc)
      use p2_gg_httbar_kinematics
      implicit none
      real(ki), dimension(num_legs, 4), intent(in) :: vecs
      real(ki), dimension(2*num_legs*num_legs) :: ampsc
      real(ki), dimension(num_legs, 4) :: pvecs
      integer :: i
      complex(ki) :: pm, mp
      complex(ki), dimension(numcs) :: heli_amp0
      complex(ki), dimension(numcs) :: heli_amp1
      complex(ki), dimension(numcs) :: heli_amp2
      complex(ki), dimension(numcs) :: heli_amp3
      complex(ki), dimension(numcs) :: heli_amp4
      complex(ki), dimension(numcs) :: heli_amp5
      complex(ki), dimension(numcs) :: heli_amp6
      complex(ki), dimension(numcs) :: heli_amp7
      complex(ki), dimension(numcs) :: heli_amp8
      complex(ki), dimension(numcs) :: heli_amp9
      complex(ki), dimension(numcs) :: heli_amp10
      complex(ki), dimension(numcs) :: heli_amp11
      complex(ki), dimension(numcs) :: heli_amp12
      complex(ki), dimension(numcs) :: heli_amp13
      complex(ki), dimension(numcs) :: heli_amp14
      complex(ki), dimension(numcs) :: heli_amp15
      complex(ki), dimension(4) :: eps1
      complex(ki), dimension(4) :: eps2

      ampsc(:) = 0.0_ki
      !---#[ Initialize helicity amplitudes :
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp0 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp1 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp2 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp3 = amplitude0l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp4 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp5 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp6 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp7 = amplitude4l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp8 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp9 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp10 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp11 = amplitude8l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, -1)
      !---#] reinitialize kinematics:
      heli_amp12 = amplitude12l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, -1)
      !---#] reinitialize kinematics:
      heli_amp13 = amplitude12l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, -1, +1)
      !---#] reinitialize kinematics:
      heli_amp14 = amplitude12l0()
      !---#[ reinitialize kinematics:
      pvecs(1,:) = vecs(1,:)
      pvecs(2,:) = vecs(2,:)
      pvecs(3,:) = vecs(3,:)
      pvecs(4,:) = vecs(4,:)
      pvecs(5,:) = vecs(5,:)
      call init_event(pvecs, +1, +1)
      !---#] reinitialize kinematics:
      heli_amp15 = amplitude12l0()
      !---#] Initialize helicity amplitudes :

       
      !---#[ pair 12 :

      mp  = 0.0_ki &
      &          + square_1_2_sc(heli_amp0,heli_amp1) &
      &          + square_1_2_sc(heli_amp2,heli_amp3) &
      &          + square_1_2_sc(heli_amp4,heli_amp5) &
      &          + square_1_2_sc(heli_amp6,heli_amp7) &
      &          + square_1_2_sc(heli_amp8,heli_amp9) &
      &          + square_1_2_sc(heli_amp10,heli_amp11) &
      &          + square_1_2_sc(heli_amp12,heli_amp13) &
      &          + square_1_2_sc(heli_amp14,heli_amp15)

      ampsc(2*(1-1)+2*(2-1)*num_legs+1)   = ampsc(2*(1-1)+2*(2-1)*num_legs +1) + real(mp, ki)
      ampsc(2*(1-1)+2*(2-1)*num_legs+2) = ampsc(2*(1-1)+2*(2-1)*num_legs + 2)  + real(aimag(mp),ki)

      !---#] pair 12 :
      
      !---#[ pair 14 :

      mp  = 0.0_ki &
      &          + square_1_4_sc(heli_amp0,heli_amp1) &
      &          + square_1_4_sc(heli_amp2,heli_amp3) &
      &          + square_1_4_sc(heli_amp4,heli_amp5) &
      &          + square_1_4_sc(heli_amp6,heli_amp7) &
      &          + square_1_4_sc(heli_amp8,heli_amp9) &
      &          + square_1_4_sc(heli_amp10,heli_amp11) &
      &          + square_1_4_sc(heli_amp12,heli_amp13) &
      &          + square_1_4_sc(heli_amp14,heli_amp15)

      ampsc(2*(1-1)+2*(4-1)*num_legs+1)   = ampsc(2*(1-1)+2*(4-1)*num_legs +1) + real(mp, ki)
      ampsc(2*(1-1)+2*(4-1)*num_legs+2) = ampsc(2*(1-1)+2*(4-1)*num_legs + 2)  + real(aimag(mp),ki)

      !---#] pair 14 :
      
      !---#[ pair 15 :

      mp  = 0.0_ki &
      &          + square_1_5_sc(heli_amp0,heli_amp1) &
      &          + square_1_5_sc(heli_amp2,heli_amp3) &
      &          + square_1_5_sc(heli_amp4,heli_amp5) &
      &          + square_1_5_sc(heli_amp6,heli_amp7) &
      &          + square_1_5_sc(heli_amp8,heli_amp9) &
      &          + square_1_5_sc(heli_amp10,heli_amp11) &
      &          + square_1_5_sc(heli_amp12,heli_amp13) &
      &          + square_1_5_sc(heli_amp14,heli_amp15)

      ampsc(2*(1-1)+2*(5-1)*num_legs+1)   = ampsc(2*(1-1)+2*(5-1)*num_legs +1) + real(mp, ki)
      ampsc(2*(1-1)+2*(5-1)*num_legs+2) = ampsc(2*(1-1)+2*(5-1)*num_legs + 2)  + real(aimag(mp),ki)

      !---#] pair 15 :
      
      !---#[ pair 21 :

      mp  = 0.0_ki &
      &          + square_2_1_sc(heli_amp0,heli_amp2) &
      &          + square_2_1_sc(heli_amp1,heli_amp3) &
      &          + square_2_1_sc(heli_amp4,heli_amp6) &
      &          + square_2_1_sc(heli_amp5,heli_amp7) &
      &          + square_2_1_sc(heli_amp8,heli_amp10) &
      &          + square_2_1_sc(heli_amp9,heli_amp11) &
      &          + square_2_1_sc(heli_amp12,heli_amp14) &
      &          + square_2_1_sc(heli_amp13,heli_amp15)

      ampsc(2*(2-1)+2*(1-1)*num_legs+1)   = ampsc(2*(2-1)+2*(1-1)*num_legs +1) + real(mp, ki)
      ampsc(2*(2-1)+2*(1-1)*num_legs+2) = ampsc(2*(2-1)+2*(1-1)*num_legs + 2)  + real(aimag(mp),ki)

      !---#] pair 21 :
       
      !---#[ pair 24 :

      mp  = 0.0_ki &
      &          + square_2_4_sc(heli_amp0,heli_amp2) &
      &          + square_2_4_sc(heli_amp1,heli_amp3) &
      &          + square_2_4_sc(heli_amp4,heli_amp6) &
      &          + square_2_4_sc(heli_amp5,heli_amp7) &
      &          + square_2_4_sc(heli_amp8,heli_amp10) &
      &          + square_2_4_sc(heli_amp9,heli_amp11) &
      &          + square_2_4_sc(heli_amp12,heli_amp14) &
      &          + square_2_4_sc(heli_amp13,heli_amp15)

      ampsc(2*(2-1)+2*(4-1)*num_legs+1)   = ampsc(2*(2-1)+2*(4-1)*num_legs +1) + real(mp, ki)
      ampsc(2*(2-1)+2*(4-1)*num_legs+2) = ampsc(2*(2-1)+2*(4-1)*num_legs + 2)  + real(aimag(mp),ki)

      !---#] pair 24 :
      
      !---#[ pair 25 :

      mp  = 0.0_ki &
      &          + square_2_5_sc(heli_amp0,heli_amp2) &
      &          + square_2_5_sc(heli_amp1,heli_amp3) &
      &          + square_2_5_sc(heli_amp4,heli_amp6) &
      &          + square_2_5_sc(heli_amp5,heli_amp7) &
      &          + square_2_5_sc(heli_amp8,heli_amp10) &
      &          + square_2_5_sc(heli_amp9,heli_amp11) &
      &          + square_2_5_sc(heli_amp12,heli_amp14) &
      &          + square_2_5_sc(heli_amp13,heli_amp15)

      ampsc(2*(2-1)+2*(5-1)*num_legs+1)   = ampsc(2*(2-1)+2*(5-1)*num_legs +1) + real(mp, ki)
      ampsc(2*(2-1)+2*(5-1)*num_legs+2) = ampsc(2*(2-1)+2*(5-1)*num_legs + 2)  + real(aimag(mp),ki)

      !---#] pair 25 :
      

      

      if (include_helicity_avg_factor) then
         ampsc = ampsc / real(in_helicities, ki)
      end if
      if (include_color_avg_factor) then
         ampsc = ampsc / incolors
      end if
      if (include_symmetry_factor) then
         ampsc = ampsc / real(symmetry_factor, ki)
      end if
   end subroutine OLP_spin_correlated_lo2
   !---#] spin correlated ME :


   !---#[ construct polarisation tensor :
   pure subroutine construct_polarization_tensor(eps1, eps2, tens)
      implicit none
      complex(ki), dimension(0:3), intent(in) :: eps1, eps2
      complex(ki), dimension(0:3,0:3), intent(out) :: tens

      integer :: mu, nu

      do mu = 0,3
         do nu = 0, 3
            tens(mu,nu) = eps1(mu) * eps2(nu)
         end do
      end do
   end  subroutine construct_polarization_tensor
   !---#] construct polarisation tensor :

   pure function square_0l_0l_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => CC
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_0l_0l_sc


    
   pure function square_1_2_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => T1T2
      
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_1_2_sc
      
   pure function square_1_4_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => T1T4
      
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_1_4_sc
      
   pure function square_1_5_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => T1T5
      
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_1_5_sc
      
   pure function square_2_1_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => T1T2
      
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_2_1_sc
       
   pure function square_2_4_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => T2T4
      
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_2_4_sc
      
   pure function square_2_5_sc(color_vector1, color_vector2) result(amp)
      use p2_gg_httbar_color, only: cmat => T2T5
      
      implicit none
      complex(ki), dimension(numcs), intent(in) :: color_vector1, color_vector2
      complex(ki) :: amp
      complex(ki), dimension(numcs) :: v1, v2

      v1 = matmul(cmat, color_vector2)
      v2 = conjg(color_vector1)
      amp = sum(v1(:) * v2(:))
   end function  square_2_5_sc
      

   

   

end module p2_gg_httbar_matrix
