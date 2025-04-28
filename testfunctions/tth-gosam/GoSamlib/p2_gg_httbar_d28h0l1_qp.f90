module     p2_gg_httbar_d28h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d28h0l1_qp.f90
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
      use p2_gg_httbar_abbrevd28h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc28(89)
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspl4
      complex(ki) :: Qspk1
      complex(ki) :: QspQ
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspl4 = dotproduct(Q,l4)
      Qspk1 = dotproduct(Q,k1)
      QspQ = dotproduct(Q,Q)
      acc28(1)=abb28(9)
      acc28(2)=abb28(10)
      acc28(3)=abb28(11)
      acc28(4)=abb28(12)
      acc28(5)=abb28(13)
      acc28(6)=abb28(14)
      acc28(7)=abb28(15)
      acc28(8)=abb28(16)
      acc28(9)=abb28(17)
      acc28(10)=abb28(18)
      acc28(11)=abb28(19)
      acc28(12)=abb28(20)
      acc28(13)=abb28(21)
      acc28(14)=abb28(22)
      acc28(15)=abb28(23)
      acc28(16)=abb28(24)
      acc28(17)=abb28(25)
      acc28(18)=abb28(26)
      acc28(19)=abb28(27)
      acc28(20)=abb28(28)
      acc28(21)=abb28(29)
      acc28(22)=abb28(30)
      acc28(23)=abb28(31)
      acc28(24)=abb28(32)
      acc28(25)=abb28(33)
      acc28(26)=abb28(34)
      acc28(27)=abb28(36)
      acc28(28)=abb28(37)
      acc28(29)=abb28(38)
      acc28(30)=abb28(39)
      acc28(31)=abb28(40)
      acc28(32)=abb28(41)
      acc28(33)=abb28(42)
      acc28(34)=abb28(43)
      acc28(35)=abb28(45)
      acc28(36)=abb28(47)
      acc28(37)=abb28(49)
      acc28(38)=abb28(50)
      acc28(39)=abb28(51)
      acc28(40)=abb28(52)
      acc28(41)=abb28(53)
      acc28(42)=abb28(54)
      acc28(43)=abb28(55)
      acc28(44)=abb28(56)
      acc28(45)=abb28(58)
      acc28(46)=abb28(59)
      acc28(47)=abb28(60)
      acc28(48)=abb28(61)
      acc28(49)=abb28(62)
      acc28(50)=abb28(68)
      acc28(51)=abb28(69)
      acc28(52)=abb28(71)
      acc28(53)=acc28(9)*Qspval3k1
      acc28(54)=acc28(20)*Qspvak2l3
      acc28(55)=acc28(24)*Qspvak1l3
      acc28(56)=acc28(25)*Qspvae1k2
      acc28(57)=acc28(31)*Qspvak1k2
      acc28(58)=acc28(34)*Qspk2
      acc28(59)=acc28(39)*Qspval5k1
      acc28(60)=acc28(40)*Qspval3l4
      acc28(61)=acc28(41)*Qspval4k2
      acc28(62)=acc28(46)*Qspval5l4
      acc28(63)=acc28(52)*Qspval4l3
      acc28(64)=Qspval5e1*acc28(43)
      acc28(65)=Qspvae1l3*acc28(50)
      acc28(66)=Qspval3e1*acc28(48)
      acc28(53)=acc28(66)+acc28(65)+acc28(64)+acc28(63)+acc28(62)+acc28(61)+acc&
      &28(60)+acc28(59)+acc28(58)+acc28(57)+acc28(56)+acc28(55)+acc28(54)+acc28&
      &(19)+acc28(53)
      acc28(53)=Qspe2*acc28(53)
      acc28(54)=acc28(5)*Qspk2
      acc28(55)=acc28(6)*Qspvak2l3
      acc28(56)=acc28(10)*Qspval3k1
      acc28(57)=acc28(14)*Qspvak1l3
      acc28(58)=acc28(16)*Qspval3l4
      acc28(59)=acc28(18)*Qspval4k2
      acc28(60)=acc28(22)*Qspvae1k2
      acc28(61)=acc28(29)*Qspvak1k2
      acc28(62)=acc28(33)*Qspval5l4
      acc28(63)=acc28(42)*Qspval5k1
      acc28(64)=acc28(51)*Qspval4l3
      acc28(65)=Qspvae2e1*acc28(35)
      acc28(66)=Qspvae1e2*acc28(21)
      acc28(67)=Qspval5e2*acc28(36)
      acc28(68)=Qspvae2l4*acc28(2)
      acc28(69)=Qspval4e2*acc28(27)
      acc28(70)=Qspvae1l4*acc28(45)
      acc28(71)=Qspval4e1*acc28(47)
      acc28(72)=Qspvae2l3*acc28(32)
      acc28(73)=Qspval3e2*acc28(49)
      acc28(74)=Qspvae2k2*acc28(4)
      acc28(75)=Qspvak2e2*acc28(3)
      acc28(76)=Qspvak2e1*acc28(30)
      acc28(77)=Qspvae2k1*acc28(37)
      acc28(78)=Qspvak1e2*acc28(38)
      acc28(79)=Qspvae1k1*acc28(23)
      acc28(80)=Qspvak1e1*acc28(26)
      acc28(81)=Qspval5k2*acc28(28)
      acc28(82)=Qspval4k1*acc28(44)
      acc28(83)=Qspval3k2*acc28(7)
      acc28(84)=Qspvak2l4*acc28(1)
      acc28(85)=Qspvak2k1*acc28(12)
      acc28(86)=Qspvak1l4*acc28(13)
      acc28(87)=Qspl4*acc28(15)
      acc28(88)=-Qspk1*acc28(11)
      acc28(89)=-QspQ*acc28(17)
      brack=acc28(8)+acc28(53)+acc28(54)+acc28(55)+acc28(56)+acc28(57)+acc28(58&
      &)+acc28(59)+acc28(60)+acc28(61)+acc28(62)+acc28(63)+acc28(64)+acc28(65)+&
      &acc28(66)+acc28(67)+acc28(68)+acc28(69)+acc28(70)+acc28(71)+acc28(72)+ac&
      &c28(73)+acc28(74)+acc28(75)+acc28(76)+acc28(77)+acc28(78)+acc28(79)+acc2&
      &8(80)+acc28(81)+acc28(82)+acc28(83)+acc28(84)+acc28(85)+acc28(86)+acc28(&
      &87)+acc28(88)+acc28(89)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d28h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd28h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d28
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d28 = 0.0_ki
      d28 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d28, ki), aimag(d28), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d28h0l1_qp
