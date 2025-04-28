module     p0_gg_gh_d1h4l1
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d1h4l1.f90
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
      use p0_gg_gh_abbrevd1h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc1(18)
      complex(ki) :: Qspvak3k2
      complex(ki) :: Qspvak3k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak3l4
      complex(ki) :: QspQ
      Qspvak3k2 = dotproduct(Q,spvak3k2)
      Qspvak3k1 = dotproduct(Q,spvak3k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak3l4 = dotproduct(Q,spvak3l4)
      QspQ = dotproduct(Q,Q)
      acc1(1)=abb1(8)
      acc1(2)=abb1(9)
      acc1(3)=abb1(10)
      acc1(4)=abb1(11)
      acc1(5)=abb1(12)
      acc1(6)=abb1(14)
      acc1(7)=abb1(15)
      acc1(8)=abb1(16)
      acc1(9)=abb1(17)
      acc1(10)=abb1(21)
      acc1(11)=abb1(22)
      acc1(12)=acc1(7)*Qspvak3k2
      acc1(13)=acc1(9)*Qspvak3k1
      acc1(12)=acc1(13)+acc1(12)+acc1(1)
      acc1(12)=Qspvak1k2*acc1(12)
      acc1(13)=acc1(10)*Qspvak3k2
      acc1(13)=acc1(13)+acc1(8)
      acc1(13)=Qspk2*acc1(13)
      acc1(14)=acc1(6)*Qspvak3k2
      acc1(15)=acc1(11)*Qspvak3k1
      acc1(16)=Qspval4k2*acc1(3)
      acc1(17)=Qspvak3l4*acc1(4)
      acc1(18)=QspQ*acc1(5)
      brack=acc1(2)+acc1(12)+acc1(13)+acc1(14)+acc1(15)+acc1(16)+acc1(17)+acc1(&
      &18)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d1h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd1h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d1
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d1 = 0.0_ki
      d1 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d1, ki), aimag(d1), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d1h4l1
