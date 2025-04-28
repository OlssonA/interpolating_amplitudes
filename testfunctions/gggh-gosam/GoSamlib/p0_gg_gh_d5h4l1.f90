module     p0_gg_gh_d5h4l1
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d5h4l1.f90
   ! generator: buildfortran.py
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd5h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc5(17)
      complex(ki) :: Qspvak3k2
      complex(ki) :: Qspvak1k3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1l4
      complex(ki) :: QspQ
      Qspvak3k2 = dotproduct(Q,spvak3k2)
      Qspvak1k3 = dotproduct(Q,spvak1k3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      QspQ = dotproduct(Q,Q)
      acc5(1)=abb5(7)
      acc5(2)=abb5(8)
      acc5(3)=abb5(9)
      acc5(4)=abb5(10)
      acc5(5)=abb5(11)
      acc5(6)=abb5(13)
      acc5(7)=abb5(14)
      acc5(8)=abb5(15)
      acc5(9)=abb5(16)
      acc5(10)=abb5(17)
      acc5(11)=acc5(5)*Qspvak3k2
      acc5(11)=acc5(11)+acc5(1)
      acc5(11)=Qspvak1k3*acc5(11)
      acc5(12)=acc5(3)*Qspvak1k2
      acc5(12)=acc5(8)+acc5(12)
      acc5(12)=Qspk2*acc5(12)
      acc5(13)=acc5(7)*Qspvak3k2
      acc5(14)=acc5(10)*Qspvak1k2
      acc5(15)=Qspval4k2*acc5(2)
      acc5(16)=Qspvak1l4*acc5(4)
      acc5(17)=QspQ*acc5(9)
      brack=acc5(6)+acc5(11)+acc5(12)+acc5(13)+acc5(14)+acc5(15)+acc5(16)+acc5(&
      &17)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d5h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd5h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d5
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k4-k3
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d5 = 0.0_ki
      d5 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d5, ki), aimag(d5), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d5h4l1
