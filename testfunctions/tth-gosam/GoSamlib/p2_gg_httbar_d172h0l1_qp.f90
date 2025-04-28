module     p2_gg_httbar_d172h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d172h0l1_qp.f90
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
      use p2_gg_httbar_abbrevd172h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc172(33)
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: QspQ
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      QspQ = dotproduct(Q,Q)
      acc172(1)=abb172(12)
      acc172(2)=abb172(13)
      acc172(3)=abb172(14)
      acc172(4)=abb172(15)
      acc172(5)=abb172(17)
      acc172(6)=abb172(21)
      acc172(7)=abb172(22)
      acc172(8)=abb172(23)
      acc172(9)=abb172(24)
      acc172(10)=abb172(25)
      acc172(11)=abb172(26)
      acc172(12)=abb172(27)
      acc172(13)=abb172(29)
      acc172(14)=abb172(30)
      acc172(15)=abb172(38)
      acc172(16)=abb172(39)
      acc172(17)=abb172(48)
      acc172(18)=abb172(54)
      acc172(19)=abb172(56)
      acc172(20)=abb172(57)
      acc172(21)=abb172(63)
      acc172(22)=abb172(66)
      acc172(23)=-Qspvae1l4*acc172(22)
      acc172(24)=Qspvae1l5*acc172(9)
      acc172(25)=Qspvae1e2*acc172(8)
      acc172(26)=Qspvae1k2*acc172(11)
      acc172(23)=acc172(26)+acc172(25)+acc172(24)+acc172(7)+acc172(23)
      acc172(23)=Qspval4e1*acc172(23)
      acc172(24)=Qspvae1l4*acc172(21)
      acc172(25)=-Qspvae1l5*acc172(22)
      acc172(26)=Qspvae1e2*acc172(16)
      acc172(27)=Qspvae1k2*acc172(14)
      acc172(24)=acc172(27)+acc172(26)+acc172(25)+acc172(15)+acc172(24)
      acc172(24)=Qspval5e1*acc172(24)
      acc172(25)=Qspvak2e1*acc172(4)
      acc172(26)=Qspvae2e1*acc172(13)
      acc172(25)=acc172(26)+acc172(12)+acc172(25)
      acc172(25)=Qspvae1k2*acc172(25)
      acc172(26)=acc172(10)*Qspvae1l3
      acc172(27)=acc172(6)*Qspval3e1
      acc172(28)=acc172(2)*QspQ
      acc172(29)=Qspvak2e1*acc172(3)
      acc172(30)=Qspvae1l4*acc172(20)
      acc172(31)=Qspvae1l5*acc172(19)
      acc172(32)=Qspvae2e1*acc172(5)
      acc172(33)=Qspvae2e1*acc172(18)
      acc172(33)=acc172(17)+acc172(33)
      acc172(33)=Qspvae1e2*acc172(33)
      brack=acc172(1)+acc172(23)+acc172(24)+acc172(25)+acc172(26)+acc172(27)+ac&
      &c172(28)+acc172(29)+acc172(30)+acc172(31)+acc172(32)+acc172(33)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d172h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd172h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d172
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d172 = 0.0_ki
      d172 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d172, ki), aimag(d172), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d172h0l1_qp
