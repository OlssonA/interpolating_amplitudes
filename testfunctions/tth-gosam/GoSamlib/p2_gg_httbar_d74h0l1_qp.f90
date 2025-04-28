module     p2_gg_httbar_d74h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d74h0l1_qp.f90
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
      use p2_gg_httbar_abbrevd74h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc74(117)
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspl4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4k2
      complex(ki) :: QspQ
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2l3
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspl4 = dotproduct(Q,l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspk2 = dotproduct(Q,k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      QspQ = dotproduct(Q,Q)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      acc74(1)=abb74(9)
      acc74(2)=abb74(10)
      acc74(3)=abb74(11)
      acc74(4)=abb74(12)
      acc74(5)=abb74(13)
      acc74(6)=abb74(14)
      acc74(7)=abb74(15)
      acc74(8)=abb74(16)
      acc74(9)=abb74(17)
      acc74(10)=abb74(18)
      acc74(11)=abb74(19)
      acc74(12)=abb74(20)
      acc74(13)=abb74(21)
      acc74(14)=abb74(22)
      acc74(15)=abb74(23)
      acc74(16)=abb74(24)
      acc74(17)=abb74(25)
      acc74(18)=abb74(26)
      acc74(19)=abb74(27)
      acc74(20)=abb74(28)
      acc74(21)=abb74(29)
      acc74(22)=abb74(30)
      acc74(23)=abb74(31)
      acc74(24)=abb74(32)
      acc74(25)=abb74(33)
      acc74(26)=abb74(34)
      acc74(27)=abb74(36)
      acc74(28)=abb74(37)
      acc74(29)=abb74(38)
      acc74(30)=abb74(39)
      acc74(31)=abb74(40)
      acc74(32)=abb74(41)
      acc74(33)=abb74(42)
      acc74(34)=abb74(43)
      acc74(35)=abb74(44)
      acc74(36)=abb74(45)
      acc74(37)=abb74(46)
      acc74(38)=abb74(47)
      acc74(39)=abb74(48)
      acc74(40)=abb74(49)
      acc74(41)=abb74(50)
      acc74(42)=abb74(51)
      acc74(43)=abb74(52)
      acc74(44)=abb74(56)
      acc74(45)=abb74(58)
      acc74(46)=abb74(59)
      acc74(47)=abb74(61)
      acc74(48)=abb74(63)
      acc74(49)=abb74(64)
      acc74(50)=abb74(66)
      acc74(51)=abb74(67)
      acc74(52)=abb74(68)
      acc74(53)=abb74(75)
      acc74(54)=abb74(77)
      acc74(55)=abb74(78)
      acc74(56)=abb74(81)
      acc74(57)=abb74(82)
      acc74(58)=abb74(90)
      acc74(59)=abb74(91)
      acc74(60)=abb74(98)
      acc74(61)=abb74(102)
      acc74(62)=abb74(103)
      acc74(63)=abb74(107)
      acc74(64)=abb74(109)
      acc74(65)=abb74(110)
      acc74(66)=abb74(112)
      acc74(67)=abb74(115)
      acc74(68)=abb74(118)
      acc74(69)=abb74(120)
      acc74(70)=abb74(121)
      acc74(71)=abb74(123)
      acc74(72)=abb74(124)
      acc74(73)=abb74(125)
      acc74(74)=abb74(126)
      acc74(75)=abb74(127)
      acc74(76)=abb74(128)
      acc74(77)=abb74(129)
      acc74(78)=abb74(130)
      acc74(79)=Qspval3k2*acc74(12)
      acc74(80)=Qspval4l3*acc74(35)
      acc74(81)=Qspl4*acc74(62)
      acc74(82)=Qspvak1k2*acc74(15)
      acc74(83)=Qspvak1l3*acc74(38)
      acc74(84)=Qspvak2l3*acc74(31)
      acc74(85)=Qspval3k1*acc74(28)
      acc74(86)=Qspval3l5*acc74(37)
      acc74(87)=Qspval4k1*acc74(25)
      acc74(88)=Qspval4l5*acc74(78)
      acc74(89)=Qspval5k2*acc74(77)
      acc74(90)=Qspval5l3*acc74(75)
      acc74(91)=Qspvae1k2*acc74(51)
      acc74(92)=Qspval3e1*acc74(66)
      acc74(93)=Qspvae1l3*acc74(65)
      acc74(94)=Qspval4e1*acc74(59)
      acc74(95)=Qspk2*acc74(18)
      acc74(96)=Qspval4k2*acc74(8)
      acc74(97)=QspQ*acc74(24)
      acc74(79)=acc74(97)+acc74(96)+acc74(95)+acc74(94)+acc74(93)+acc74(92)+acc&
      &74(91)+acc74(90)+acc74(89)+acc74(88)+acc74(87)+acc74(86)+acc74(85)+acc74&
      &(84)+acc74(83)+acc74(82)+acc74(81)+acc74(80)+acc74(17)+acc74(79)
      acc74(79)=Qspe2*acc74(79)
      acc74(80)=acc74(72)*Qspvae2k1
      acc74(81)=-acc74(53)*Qspvae2e1
      acc74(82)=-acc74(52)*Qspvae1e2
      acc74(83)=-acc74(47)*Qspvae2l5
      acc74(84)=acc74(46)*Qspval5e2
      acc74(85)=-acc74(45)*Qspvak1e2
      acc74(86)=-Qspvae2k2*acc74(68)
      acc74(87)=Qspvak2e2*acc74(69)
      acc74(88)=Qspval4e2*acc74(39)
      acc74(80)=acc74(88)+acc74(87)+acc74(86)+acc74(85)+acc74(84)+acc74(83)+acc&
      &74(82)+acc74(81)+acc74(21)+acc74(80)
      acc74(80)=QspQ*acc74(80)
      acc74(81)=Qspvak1k2*acc74(13)
      acc74(82)=Qspval5k2*acc74(48)
      acc74(83)=Qspvae1k2*acc74(42)
      acc74(84)=Qspk2*acc74(67)
      acc74(85)=Qspval4k2*acc74(10)
      acc74(81)=acc74(85)+acc74(84)+acc74(83)+acc74(82)+acc74(22)+acc74(81)
      acc74(81)=Qspvak2e2*acc74(81)
      acc74(82)=-Qspl4*acc74(23)
      acc74(83)=acc74(69)*Qspvak2l4
      acc74(84)=-acc74(52)*Qspvae1l4
      acc74(85)=acc74(46)*Qspval5l4
      acc74(86)=-acc74(45)*Qspvak1l4
      acc74(82)=acc74(86)+acc74(85)+acc74(84)+acc74(83)+acc74(26)+acc74(82)
      acc74(82)=Qspval4e2*acc74(82)
      acc74(83)=acc74(69)*Qspvak2l3
      acc74(84)=-acc74(52)*Qspvae1l3
      acc74(85)=acc74(46)*Qspval5l3
      acc74(86)=-acc74(45)*Qspvak1l3
      acc74(83)=acc74(86)+acc74(85)+acc74(84)+acc74(64)+acc74(83)
      acc74(83)=Qspval3e2*acc74(83)
      acc74(84)=acc74(72)*Qspval4k1
      acc74(85)=-acc74(53)*Qspval4e1
      acc74(86)=-acc74(47)*Qspval4l5
      acc74(87)=-Qspval4k2*acc74(68)
      acc74(84)=acc74(87)+acc74(86)+acc74(85)+acc74(57)+acc74(84)
      acc74(84)=Qspvae2l4*acc74(84)
      acc74(85)=Qspvak2k1*acc74(14)
      acc74(86)=Qspvak2l5*acc74(36)
      acc74(87)=Qspvak2e1*acc74(60)
      acc74(88)=Qspk2*acc74(1)
      acc74(85)=acc74(88)+acc74(87)+acc74(86)+acc74(34)+acc74(85)
      acc74(85)=Qspvae2k2*acc74(85)
      acc74(86)=acc74(72)*Qspval3k1
      acc74(87)=-acc74(53)*Qspval3e1
      acc74(88)=-acc74(47)*Qspval3l5
      acc74(86)=acc74(88)+acc74(87)+acc74(56)+acc74(86)
      acc74(86)=Qspvae2l3*acc74(86)
      acc74(87)=Qspvak1l4*acc74(27)
      acc74(88)=Qspvak2k1*acc74(3)
      acc74(89)=Qspvak2l4*acc74(2)
      acc74(90)=Qspvak2l5*acc74(30)
      acc74(91)=Qspval3k2*acc74(5)
      acc74(92)=Qspval4l3*acc74(11)
      acc74(93)=Qspval5l4*acc74(74)
      acc74(94)=Qspvak1e2*acc74(6)
      acc74(95)=Qspvae2k1*acc74(73)
      acc74(96)=Qspvak2e1*acc74(71)
      acc74(97)=Qspvae1l4*acc74(44)
      acc74(98)=Qspval5e2*acc74(55)
      acc74(99)=Qspvae2l5*acc74(54)
      acc74(100)=Qspvae1e2*acc74(33)
      acc74(101)=Qspvae2e1*acc74(40)
      acc74(102)=Qspl4*acc74(19)
      acc74(103)=Qspvak1k2*acc74(43)
      acc74(104)=Qspvak1l3*acc74(32)
      acc74(105)=Qspvak2l3*acc74(9)
      acc74(106)=Qspval3k1*acc74(16)
      acc74(107)=Qspval3l5*acc74(29)
      acc74(108)=Qspval4k1*acc74(20)
      acc74(109)=Qspval4l5*acc74(63)
      acc74(110)=Qspval5k2*acc74(76)
      acc74(111)=Qspval5l3*acc74(49)
      acc74(112)=Qspvae1k2*acc74(70)
      acc74(113)=Qspval3e1*acc74(50)
      acc74(114)=Qspvae1l3*acc74(61)
      acc74(115)=Qspval4e1*acc74(58)
      acc74(116)=Qspk2*acc74(41)
      acc74(117)=Qspval4k2*acc74(4)
      brack=acc74(7)+acc74(79)+acc74(80)+acc74(81)+acc74(82)+acc74(83)+acc74(84&
      &)+acc74(85)+acc74(86)+acc74(87)+acc74(88)+acc74(89)+acc74(90)+acc74(91)+&
      &acc74(92)+acc74(93)+acc74(94)+acc74(95)+acc74(96)+acc74(97)+acc74(98)+ac&
      &c74(99)+acc74(100)+acc74(101)+acc74(102)+acc74(103)+acc74(104)+acc74(105&
      &)+acc74(106)+acc74(107)+acc74(108)+acc74(109)+acc74(110)+acc74(111)+acc7&
      &4(112)+acc74(113)+acc74(114)+acc74(115)+acc74(116)+acc74(117)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d74h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd74h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d74
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d74 = 0.0_ki
      d74 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d74, ki), aimag(d74), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d74h0l1_qp
