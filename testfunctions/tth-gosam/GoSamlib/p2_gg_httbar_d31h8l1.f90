module     p2_gg_httbar_d31h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d31h8l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd31h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc31(89)
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspl5
      complex(ki) :: Qspk2
      complex(ki) :: Qspk1
      complex(ki) :: QspQ
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspl5 = dotproduct(Q,l5)
      Qspk2 = dotproduct(Q,k2)
      Qspk1 = dotproduct(Q,k1)
      QspQ = dotproduct(Q,Q)
      acc31(1)=abb31(9)
      acc31(2)=abb31(10)
      acc31(3)=abb31(11)
      acc31(4)=abb31(12)
      acc31(5)=abb31(13)
      acc31(6)=abb31(14)
      acc31(7)=abb31(15)
      acc31(8)=abb31(16)
      acc31(9)=abb31(17)
      acc31(10)=abb31(18)
      acc31(11)=abb31(19)
      acc31(12)=abb31(20)
      acc31(13)=abb31(21)
      acc31(14)=abb31(22)
      acc31(15)=abb31(23)
      acc31(16)=abb31(25)
      acc31(17)=abb31(26)
      acc31(18)=abb31(27)
      acc31(19)=abb31(28)
      acc31(20)=abb31(29)
      acc31(21)=abb31(30)
      acc31(22)=abb31(31)
      acc31(23)=abb31(32)
      acc31(24)=abb31(33)
      acc31(25)=abb31(34)
      acc31(26)=abb31(35)
      acc31(27)=abb31(36)
      acc31(28)=abb31(37)
      acc31(29)=abb31(39)
      acc31(30)=abb31(40)
      acc31(31)=abb31(41)
      acc31(32)=abb31(42)
      acc31(33)=abb31(43)
      acc31(34)=abb31(44)
      acc31(35)=abb31(45)
      acc31(36)=abb31(46)
      acc31(37)=abb31(47)
      acc31(38)=abb31(48)
      acc31(39)=abb31(49)
      acc31(40)=abb31(50)
      acc31(41)=abb31(51)
      acc31(42)=abb31(52)
      acc31(43)=abb31(53)
      acc31(44)=abb31(55)
      acc31(45)=abb31(56)
      acc31(46)=abb31(57)
      acc31(47)=abb31(58)
      acc31(48)=abb31(59)
      acc31(49)=abb31(60)
      acc31(50)=abb31(63)
      acc31(51)=abb31(68)
      acc31(52)=abb31(72)
      acc31(53)=acc31(11)*Qspvak1k2
      acc31(54)=acc31(21)*Qspval5k2
      acc31(55)=acc31(28)*Qspval3l5
      acc31(56)=acc31(30)*Qspval3k2
      acc31(57)=acc31(33)*Qspval4k2
      acc31(58)=acc31(38)*Qspval4l5
      acc31(59)=acc31(41)*Qspval3k1
      acc31(60)=acc31(42)*Qspval4k1
      acc31(61)=acc31(45)*Qspvae1k2
      acc31(62)=acc31(47)*Qspval5l3
      acc31(63)=acc31(48)*Qspvak1l3
      acc31(64)=Qspval4e1*acc31(51)
      acc31(65)=Qspvae1l3*acc31(29)
      acc31(66)=Qspval3e1*acc31(34)
      acc31(53)=acc31(66)+acc31(65)+acc31(64)+acc31(63)+acc31(62)+acc31(61)+acc&
      &31(60)+acc31(59)+acc31(58)+acc31(57)+acc31(56)+acc31(55)+acc31(54)+acc31&
      &(53)+acc31(10)
      acc31(53)=Qspe2*acc31(53)
      acc31(54)=acc31(1)*Qspvak1k2
      acc31(55)=acc31(3)*Qspval4k2
      acc31(56)=acc31(15)*Qspval3k2
      acc31(57)=acc31(16)*Qspval4k1
      acc31(58)=acc31(17)*Qspval5k2
      acc31(59)=acc31(18)*Qspval4l5
      acc31(60)=acc31(20)*Qspvae1k2
      acc31(61)=acc31(22)*Qspval3l5
      acc31(62)=acc31(36)*Qspval5l3
      acc31(63)=acc31(40)*Qspval3k1
      acc31(64)=acc31(46)*Qspvak1l3
      acc31(65)=Qspvae2e1*acc31(4)
      acc31(66)=Qspvae1e2*acc31(13)
      acc31(67)=Qspvae2l5*acc31(23)
      acc31(68)=Qspval5e2*acc31(37)
      acc31(69)=Qspvae1l5*acc31(49)
      acc31(70)=Qspval5e1*acc31(50)
      acc31(71)=Qspval4e2*acc31(26)
      acc31(72)=Qspvae2l3*acc31(52)
      acc31(73)=Qspval3e2*acc31(31)
      acc31(74)=Qspvae2k2*acc31(6)
      acc31(75)=Qspvak2e2*acc31(7)
      acc31(76)=Qspvak2e1*acc31(12)
      acc31(77)=Qspvae2k1*acc31(25)
      acc31(78)=Qspvak1e2*acc31(32)
      acc31(79)=Qspvae1k1*acc31(14)
      acc31(80)=Qspvak1e1*acc31(35)
      acc31(81)=Qspval5k1*acc31(5)
      acc31(82)=Qspvak2l5*acc31(44)
      acc31(83)=Qspvak2l3*acc31(24)
      acc31(84)=Qspvak2k1*acc31(27)
      acc31(85)=Qspvak1l5*acc31(43)
      acc31(86)=Qspl5*acc31(8)
      acc31(87)=Qspk2*acc31(2)
      acc31(88)=-Qspk1*acc31(39)
      acc31(89)=QspQ*acc31(19)
      brack=acc31(9)+acc31(53)+acc31(54)+acc31(55)+acc31(56)+acc31(57)+acc31(58&
      &)+acc31(59)+acc31(60)+acc31(61)+acc31(62)+acc31(63)+acc31(64)+acc31(65)+&
      &acc31(66)+acc31(67)+acc31(68)+acc31(69)+acc31(70)+acc31(71)+acc31(72)+ac&
      &c31(73)+acc31(74)+acc31(75)+acc31(76)+acc31(77)+acc31(78)+acc31(79)+acc3&
      &1(80)+acc31(81)+acc31(82)+acc31(83)+acc31(84)+acc31(85)+acc31(86)+acc31(&
      &87)+acc31(88)+acc31(89)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d31h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd31h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d31
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d31 = 0.0_ki
      d31 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d31, ki), aimag(d31), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d31h8l1
