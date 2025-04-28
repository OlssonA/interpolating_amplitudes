module     p0_gg_gh_d3h0l1
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity0d3h0l1.f90
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
      use p0_gg_gh_abbrevd3h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc3(21)
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk3
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      complex(ki) :: Qspval4k3
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak1k3
      complex(ki) :: Qspvak2k1
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk3 = dotproduct(Q,k3)
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      Qspval4k3 = dotproduct(Q,spval4k3)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak1k3 = dotproduct(Q,spvak1k3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      acc3(1)=abb3(9)
      acc3(2)=abb3(10)
      acc3(3)=abb3(11)
      acc3(4)=abb3(12)
      acc3(5)=abb3(13)
      acc3(6)=abb3(15)
      acc3(7)=abb3(16)
      acc3(8)=abb3(17)
      acc3(9)=abb3(18)
      acc3(10)=abb3(19)
      acc3(11)=abb3(20)
      acc3(12)=abb3(21)
      acc3(13)=abb3(22)
      acc3(14)=acc3(12)*Qspvak1k2
      acc3(15)=-Qspk3*acc3(9)
      acc3(16)=Qspvak2k3*acc3(2)
      acc3(14)=acc3(16)+acc3(15)+acc3(14)+acc3(3)
      acc3(14)=Qspvak2k3*acc3(14)
      acc3(15)=acc3(13)*Qspk2
      acc3(16)=acc3(11)*QspQ
      acc3(17)=acc3(10)*Qspval4k3
      acc3(18)=-acc3(5)*Qspvak2l4
      acc3(19)=Qspk3*acc3(7)
      acc3(20)=Qspvak1k3*acc3(6)
      acc3(21)=Qspvak1k3*acc3(8)
      acc3(21)=acc3(4)+acc3(21)
      acc3(21)=Qspvak2k1*acc3(21)
      brack=acc3(1)+acc3(14)+acc3(15)+acc3(16)+acc3(17)+acc3(18)+acc3(19)+acc3(&
      &20)+acc3(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d3h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd3h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d3
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d3 = 0.0_ki
      d3 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d3, ki), aimag(d3), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d3h0l1
