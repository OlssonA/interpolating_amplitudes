!
module     p0_ubaru_httbar_ninjah4_qp
   ! This file has been generated for ninja
   use quadninjago_module, only: ki_nin
   use p0_ubaru_httbar_config
   implicit none
   private
   public :: ninja_reduce_group0
   public :: ninja_reduce_group1
   public :: ninja_reduce_group2
   public :: ninja_reduce_group3
contains
!---#[ reduce groups with ninja:
!-----#[ subroutine ninja_reduce_group0:
subroutine     ninja_reduce_group0(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use quadninjago_module
   use p0_ubaru_httbar_kinematics_qp
   use p0_ubaru_httbar_model_qp
   use p0_ubaru_httbar_d1h4l1_qp, only: numerator_diagram1 => numerator_ninja
   use p0_ubaru_httbar_d1h4l121_qp, only: numerator_tmu_diagram1 => numerator_tmu
   use p0_ubaru_httbar_d1h4l131_qp, only: numerator_t3_diagram1 => numerator_t3
   use p0_ubaru_httbar_d1h4l132_qp, only: numerator_t2_diagram1 => numerator_t2
   use p0_ubaru_httbar_d13h4l1_qp, only: numerator_diagram13 => numerator_ninja
   use p0_ubaru_httbar_d13h4l121_qp, only: numerator_tmu_diagram13 => numerator_tmu
   use p0_ubaru_httbar_d13h4l131_qp, only: numerator_t3_diagram13 => numerator_t3
   use p0_ubaru_httbar_d13h4l132_qp, only: numerator_t2_diagram13 => numerator_t2
   use p0_ubaru_httbar_d43h4l1_qp, only: numerator_diagram43 => numerator_ninja
   use p0_ubaru_httbar_d43h4l121_qp, only: numerator_tmu_diagram43 => numerator_tmu
   use p0_ubaru_httbar_d43h4l131_qp, only: numerator_t3_diagram43 => numerator_t3
   use p0_ubaru_httbar_d43h4l132_qp, only: numerator_t2_diagram43 => numerator_t2
   use p0_ubaru_httbar_d58h4l1_qp, only: numerator_diagram58 => numerator_ninja
   use p0_ubaru_httbar_d58h4l121_qp, only: numerator_tmu_diagram58 => numerator_tmu
   use p0_ubaru_httbar_d58h4l131_qp, only: numerator_t3_diagram58 => numerator_t3
   use p0_ubaru_httbar_d58h4l132_qp, only: numerator_t2_diagram58 => numerator_t2
   use p0_ubaru_httbar_globalsl1_qp, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(4,4) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(4) :: msq
   real(ki_nin), dimension(4,4) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(k5, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k4, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k4-k3, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(-mT**2+es34, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='1'>"
         end if
         call quadninja_diagram(numerator_diagram1, numerator_t3_diagram1, nume&
         &rator_t2_diagram1, &
          &  numerator_tmu_diagram1, &
          & 4, 3, 2, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='13'>"
         end if
         call quadninja_diagram(numerator_diagram13, numerator_t3_diagram13, nu&
         &merator_t2_diagram13, &
          &  numerator_tmu_diagram13, &
          & 4, 3, 3, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='43'>"
         end if
         call quadninja_diagram(numerator_diagram43, numerator_t3_diagram43, nu&
         &merator_t2_diagram43, &
          &  numerator_tmu_diagram43, &
          & 4, 2, 2, (/1,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='58'>"
         end if
         call quadninja_diagram(numerator_diagram58, numerator_t3_diagram58, nu&
         &merator_t2_diagram58, &
          &  numerator_tmu_diagram58, &
          & 4, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group0
!-----#] subroutine ninja_reduce_group0:
!-----#[ subroutine ninja_reduce_group1:
subroutine     ninja_reduce_group1(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use quadninjago_module
   use p0_ubaru_httbar_kinematics_qp
   use p0_ubaru_httbar_model_qp
   use p0_ubaru_httbar_d3h4l1_qp, only: numerator_diagram3 => numerator_ninja
   use p0_ubaru_httbar_d3h4l121_qp, only: numerator_tmu_diagram3 => numerator_tmu
   use p0_ubaru_httbar_d3h4l131_qp, only: numerator_t3_diagram3 => numerator_t3
   use p0_ubaru_httbar_d3h4l132_qp, only: numerator_t2_diagram3 => numerator_t2
   use p0_ubaru_httbar_d59h4l1_qp, only: numerator_diagram59 => numerator_ninja
   use p0_ubaru_httbar_d59h4l121_qp, only: numerator_tmu_diagram59 => numerator_tmu
   use p0_ubaru_httbar_d59h4l131_qp, only: numerator_t3_diagram59 => numerator_t3
   use p0_ubaru_httbar_d59h4l132_qp, only: numerator_t2_diagram59 => numerator_t2
   use p0_ubaru_httbar_globalsl1_qp, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 3

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(4,4) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(4) :: msq
   real(ki_nin), dimension(4,4) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(-k4, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(k5, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(k5+k3, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(mT**2+es12-es34-es45+mH**2, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='3'>"
         end if
         call quadninja_diagram(numerator_diagram3, numerator_t3_diagram3, nume&
         &rator_t2_diagram3, &
          &  numerator_tmu_diagram3, &
          & 4, 3, 2, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='59'>"
         end if
         call quadninja_diagram(numerator_diagram59, numerator_t3_diagram59, nu&
         &merator_t2_diagram59, &
          &  numerator_tmu_diagram59, &
          & 4, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group1
!-----#] subroutine ninja_reduce_group1:
!-----#[ subroutine ninja_reduce_group2:
subroutine     ninja_reduce_group2(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use quadninjago_module
   use p0_ubaru_httbar_kinematics_qp
   use p0_ubaru_httbar_model_qp
   use p0_ubaru_httbar_d64h4l1_qp, only: numerator_diagram64 => numerator_ninja
   use p0_ubaru_httbar_d64h4l121_qp, only: numerator_tmu_diagram64 => numerator_tmu
   use p0_ubaru_httbar_d64h4l131_qp, only: numerator_t3_diagram64 => numerator_t3
   use p0_ubaru_httbar_d64h4l132_qp, only: numerator_t2_diagram64 => numerator_t2
   use p0_ubaru_httbar_d67h4l1_qp, only: numerator_diagram67 => numerator_ninja
   use p0_ubaru_httbar_d67h4l121_qp, only: numerator_tmu_diagram67 => numerator_tmu
   use p0_ubaru_httbar_d67h4l131_qp, only: numerator_t3_diagram67 => numerator_t3
   use p0_ubaru_httbar_d67h4l132_qp, only: numerator_t2_diagram67 => numerator_t2
   use p0_ubaru_httbar_d84h4l1_qp, only: numerator_diagram84 => numerator_ninja
   use p0_ubaru_httbar_d84h4l121_qp, only: numerator_tmu_diagram84 => numerator_tmu
   use p0_ubaru_httbar_d84h4l131_qp, only: numerator_t3_diagram84 => numerator_t3
   use p0_ubaru_httbar_d84h4l132_qp, only: numerator_t2_diagram84 => numerator_t2
   use p0_ubaru_httbar_globalsl1_qp, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 3

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = 0.0_ki_nin
   Vi(:,1) = real(-k3-k4-k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(-k3-k4, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k4, ki_nin)
   msq(4) = 0.0_ki_nin
   Vi(:,4) = real(0, ki_nin)
   msq(5) = 0.0_ki_nin
   Vi(:,5) = real(-k2, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(0.0_ki, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(0.0_ki, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es34-mT**2, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(-mT**2+es51, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(0.0_ki, ki_nin)
   s_mat(4,5) = real(0.0_ki, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(0.0_ki, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='64'>"
         end if
         call quadninja_diagram(numerator_diagram64, numerator_t3_diagram64, nu&
         &merator_t2_diagram64, &
          &  numerator_tmu_diagram64, &
          & 5, 4, 2, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='67'>"
         end if
         call quadninja_diagram(numerator_diagram67, numerator_t3_diagram67, nu&
         &merator_t2_diagram67, &
          &  numerator_tmu_diagram67, &
          & 5, 4, 2, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='84'>"
         end if
         call quadninja_diagram(numerator_diagram84, numerator_t3_diagram84, nu&
         &merator_t2_diagram84, &
          &  numerator_tmu_diagram84, &
          & 5, 5, 3, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group2
!-----#] subroutine ninja_reduce_group2:
!-----#[ subroutine ninja_reduce_group3:
subroutine     ninja_reduce_group3(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use quadninjago_module
   use p0_ubaru_httbar_kinematics_qp
   use p0_ubaru_httbar_model_qp
   use p0_ubaru_httbar_d2h4l1_qp, only: numerator_diagram2 => numerator_ninja
   use p0_ubaru_httbar_d2h4l121_qp, only: numerator_tmu_diagram2 => numerator_tmu
   use p0_ubaru_httbar_d2h4l131_qp, only: numerator_t3_diagram2 => numerator_t3
   use p0_ubaru_httbar_d2h4l132_qp, only: numerator_t2_diagram2 => numerator_t2
   use p0_ubaru_httbar_d4h4l1_qp, only: numerator_diagram4 => numerator_ninja
   use p0_ubaru_httbar_d4h4l121_qp, only: numerator_tmu_diagram4 => numerator_tmu
   use p0_ubaru_httbar_d4h4l131_qp, only: numerator_t3_diagram4 => numerator_t3
   use p0_ubaru_httbar_d4h4l132_qp, only: numerator_t2_diagram4 => numerator_t2
   use p0_ubaru_httbar_d21h4l1_qp, only: numerator_diagram21 => numerator_ninja
   use p0_ubaru_httbar_d21h4l121_qp, only: numerator_tmu_diagram21 => numerator_tmu
   use p0_ubaru_httbar_d21h4l131_qp, only: numerator_t3_diagram21 => numerator_t3
   use p0_ubaru_httbar_d21h4l132_qp, only: numerator_t2_diagram21 => numerator_t2
   use p0_ubaru_httbar_d22h4l1_qp, only: numerator_diagram22 => numerator_ninja
   use p0_ubaru_httbar_d22h4l121_qp, only: numerator_tmu_diagram22 => numerator_tmu
   use p0_ubaru_httbar_d22h4l131_qp, only: numerator_t3_diagram22 => numerator_t3
   use p0_ubaru_httbar_d22h4l132_qp, only: numerator_t2_diagram22 => numerator_t2
   use p0_ubaru_httbar_d39h4l1_qp, only: numerator_diagram39 => numerator_ninja
   use p0_ubaru_httbar_d39h4l121_qp, only: numerator_tmu_diagram39 => numerator_tmu
   use p0_ubaru_httbar_d39h4l131_qp, only: numerator_t3_diagram39 => numerator_t3
   use p0_ubaru_httbar_d39h4l132_qp, only: numerator_t2_diagram39 => numerator_t2
   use p0_ubaru_httbar_d57h4l1_qp, only: numerator_diagram57 => numerator_ninja
   use p0_ubaru_httbar_d57h4l121_qp, only: numerator_tmu_diagram57 => numerator_tmu
   use p0_ubaru_httbar_d57h4l131_qp, only: numerator_t3_diagram57 => numerator_t3
   use p0_ubaru_httbar_d57h4l132_qp, only: numerator_t2_diagram57 => numerator_t2
   use p0_ubaru_httbar_d65h4l1_qp, only: numerator_diagram65 => numerator_ninja
   use p0_ubaru_httbar_d65h4l121_qp, only: numerator_tmu_diagram65 => numerator_tmu
   use p0_ubaru_httbar_d65h4l131_qp, only: numerator_t3_diagram65 => numerator_t3
   use p0_ubaru_httbar_d65h4l132_qp, only: numerator_t2_diagram65 => numerator_t2
   use p0_ubaru_httbar_d66h4l1_qp, only: numerator_diagram66 => numerator_ninja
   use p0_ubaru_httbar_d66h4l121_qp, only: numerator_tmu_diagram66 => numerator_tmu
   use p0_ubaru_httbar_d66h4l131_qp, only: numerator_t3_diagram66 => numerator_t3
   use p0_ubaru_httbar_d66h4l132_qp, only: numerator_t2_diagram66 => numerator_t2
   use p0_ubaru_httbar_d71h4l1_qp, only: numerator_diagram71 => numerator_ninja
   use p0_ubaru_httbar_d71h4l121_qp, only: numerator_tmu_diagram71 => numerator_tmu
   use p0_ubaru_httbar_d71h4l131_qp, only: numerator_t3_diagram71 => numerator_t3
   use p0_ubaru_httbar_d71h4l132_qp, only: numerator_t2_diagram71 => numerator_t2
   use p0_ubaru_httbar_d72h4l1_qp, only: numerator_diagram72 => numerator_ninja
   use p0_ubaru_httbar_d72h4l121_qp, only: numerator_tmu_diagram72 => numerator_tmu
   use p0_ubaru_httbar_d72h4l131_qp, only: numerator_t3_diagram72 => numerator_t3
   use p0_ubaru_httbar_d72h4l132_qp, only: numerator_t2_diagram72 => numerator_t2
   use p0_ubaru_httbar_d77h4l1_qp, only: numerator_diagram77 => numerator_ninja
   use p0_ubaru_httbar_d77h4l121_qp, only: numerator_tmu_diagram77 => numerator_tmu
   use p0_ubaru_httbar_d77h4l131_qp, only: numerator_t3_diagram77 => numerator_t3
   use p0_ubaru_httbar_d77h4l132_qp, only: numerator_t2_diagram77 => numerator_t2
   use p0_ubaru_httbar_d83h4l1_qp, only: numerator_diagram83 => numerator_ninja
   use p0_ubaru_httbar_d83h4l121_qp, only: numerator_tmu_diagram83 => numerator_tmu
   use p0_ubaru_httbar_d83h4l131_qp, only: numerator_t3_diagram83 => numerator_t3
   use p0_ubaru_httbar_d83h4l132_qp, only: numerator_t2_diagram83 => numerator_t2
   use p0_ubaru_httbar_globalsl1_qp, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = 0.0_ki_nin
   Vi(:,1) = real(-k3-k4-k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(-k3-k5, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k5, ki_nin)
   msq(4) = 0.0_ki_nin
   Vi(:,4) = real(0, ki_nin)
   msq(5) = 0.0_ki_nin
   Vi(:,5) = real(-k2, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(0.0_ki, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(es34-mT**2, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(0.0_ki, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(mT**2-es45+es23-es51, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es34-es51-es12, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(0.0_ki, ki_nin)
   s_mat(4,5) = real(0.0_ki, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(0.0_ki, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='2'>"
         end if
         call quadninja_diagram(numerator_diagram2, numerator_t3_diagram2, nume&
         &rator_t2_diagram2, &
          &  numerator_tmu_diagram2, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='4'>"
         end if
         call quadninja_diagram(numerator_diagram4, numerator_t3_diagram4, nume&
         &rator_t2_diagram4, &
          &  numerator_tmu_diagram4, &
          & 5, 3, 2, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='21'>"
         end if
         call quadninja_diagram(numerator_diagram21, numerator_t3_diagram21, nu&
         &merator_t2_diagram21, &
          &  numerator_tmu_diagram21, &
          & 5, 2, 2, (/1,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='22'>"
         end if
         call quadninja_diagram(numerator_diagram22, numerator_t3_diagram22, nu&
         &merator_t2_diagram22, &
          &  numerator_tmu_diagram22, &
          & 5, 2, 2, (/2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='39'>"
         end if
         call quadninja_diagram(numerator_diagram39, numerator_t3_diagram39, nu&
         &merator_t2_diagram39, &
          &  numerator_tmu_diagram39, &
          & 5, 2, 2, (/1,4/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  +  acc
         totr = totr  +  accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='57'>"
         end if
         call quadninja_diagram(numerator_diagram57, numerator_t3_diagram57, nu&
         &merator_t2_diagram57, &
          &  numerator_tmu_diagram57, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='65'>"
         end if
         call quadninja_diagram(numerator_diagram65, numerator_t3_diagram65, nu&
         &merator_t2_diagram65, &
          &  numerator_tmu_diagram65, &
          & 5, 4, 2, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='66'>"
         end if
         call quadninja_diagram(numerator_diagram66, numerator_t3_diagram66, nu&
         &merator_t2_diagram66, &
          &  numerator_tmu_diagram66, &
          & 5, 4, 2, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='71'>"
         end if
         call quadninja_diagram(numerator_diagram71, numerator_t3_diagram71, nu&
         &merator_t2_diagram71, &
          &  numerator_tmu_diagram71, &
          & 5, 3, 2, (/2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='72'>"
         end if
         call quadninja_diagram(numerator_diagram72, numerator_t3_diagram72, nu&
         &merator_t2_diagram72, &
          &  numerator_tmu_diagram72, &
          & 5, 3, 2, (/1,2,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='77'>"
         end if
         call quadninja_diagram(numerator_diagram77, numerator_t3_diagram77, nu&
         &merator_t2_diagram77, &
          &  numerator_tmu_diagram77, &
          & 5, 3, 2, (/1,4,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='83'>"
         end if
         call quadninja_diagram(numerator_diagram83, numerator_t3_diagram83, nu&
         &merator_t2_diagram83, &
          &  numerator_tmu_diagram83, &
          & 5, 5, 3, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group3
!-----#] subroutine ninja_reduce_group3:
!---#] reduce groups with ninja:
end module p0_ubaru_httbar_ninjah4_qp
