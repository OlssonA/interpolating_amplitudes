module     p2_gg_httbar_d27h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d27h8l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd27h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc27(35)
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval3e2
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      acc27(1)=abb27(10)
      acc27(2)=abb27(11)
      acc27(3)=abb27(12)
      acc27(4)=abb27(13)
      acc27(5)=abb27(14)
      acc27(6)=abb27(15)
      acc27(7)=abb27(16)
      acc27(8)=abb27(17)
      acc27(9)=abb27(18)
      acc27(10)=abb27(19)
      acc27(11)=abb27(20)
      acc27(12)=abb27(21)
      acc27(13)=abb27(22)
      acc27(14)=abb27(23)
      acc27(15)=abb27(24)
      acc27(16)=abb27(26)
      acc27(17)=abb27(33)
      acc27(18)=abb27(34)
      acc27(19)=abb27(35)
      acc27(20)=abb27(38)
      acc27(21)=abb27(39)
      acc27(22)=abb27(40)
      acc27(23)=abb27(42)
      acc27(24)=abb27(61)
      acc27(25)=abb27(63)
      acc27(26)=acc27(4)*Qspvae2l3
      acc27(27)=acc27(7)*Qspvae2k1
      acc27(28)=-acc27(10)*Qspvae2e1
      acc27(29)=acc27(11)*Qspvae2l4
      acc27(30)=acc27(13)*Qspvae2l5
      acc27(26)=acc27(30)+acc27(29)+acc27(28)+acc27(27)+acc27(26)+acc27(1)
      acc27(26)=Qspvak2e2*acc27(26)
      acc27(27)=acc27(16)*Qspval4e2
      acc27(28)=acc27(18)*Qspvae1e2
      acc27(29)=-acc27(21)*Qspvak1e2
      acc27(27)=acc27(29)+acc27(28)+acc27(27)+acc27(9)
      acc27(27)=Qspvae2l5*acc27(27)
      acc27(28)=acc27(22)*Qspvae1e2
      acc27(29)=acc27(24)*Qspval4e2
      acc27(30)=-acc27(25)*Qspvak1e2
      acc27(28)=acc27(30)+acc27(29)+acc27(28)+acc27(20)
      acc27(28)=Qspvae2l3*acc27(28)
      acc27(29)=acc27(6)*Qspvae2k1
      acc27(30)=-acc27(17)*Qspvae2e1
      acc27(31)=-acc27(19)*Qspvae2l4
      acc27(29)=acc27(31)+acc27(30)+acc27(12)+acc27(29)
      acc27(29)=Qspval3e2*acc27(29)
      acc27(30)=acc27(3)*Qspvae2e1
      acc27(31)=acc27(5)*Qspvae2k1
      acc27(32)=acc27(8)*Qspvae1e2
      acc27(33)=acc27(14)*Qspvae2l4
      acc27(34)=acc27(15)*Qspvak1e2
      acc27(35)=acc27(23)*Qspval4e2
      brack=acc27(2)+acc27(26)+acc27(27)+acc27(28)+acc27(29)+acc27(30)+acc27(31&
      &)+acc27(32)+acc27(33)+acc27(34)+acc27(35)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d27h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd27h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d27
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k3+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d27 = 0.0_ki
      d27 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d27, ki), aimag(d27), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d27h8l1_qp
