module     p2_gg_httbar_d253h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d253h0l1.f90
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
      use p2_gg_httbar_abbrevd253h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc253(69)
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval3e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2e1
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      QspQ = dotproduct(Q,Q)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      acc253(1)=abb253(7)
      acc253(2)=abb253(8)
      acc253(3)=abb253(9)
      acc253(4)=abb253(10)
      acc253(5)=abb253(11)
      acc253(6)=abb253(12)
      acc253(7)=abb253(13)
      acc253(8)=abb253(14)
      acc253(9)=abb253(15)
      acc253(10)=abb253(16)
      acc253(11)=abb253(17)
      acc253(12)=abb253(18)
      acc253(13)=abb253(19)
      acc253(14)=abb253(20)
      acc253(15)=abb253(21)
      acc253(16)=abb253(22)
      acc253(17)=abb253(23)
      acc253(18)=abb253(24)
      acc253(19)=abb253(25)
      acc253(20)=abb253(26)
      acc253(21)=abb253(27)
      acc253(22)=abb253(28)
      acc253(23)=abb253(29)
      acc253(24)=abb253(30)
      acc253(25)=abb253(31)
      acc253(26)=abb253(32)
      acc253(27)=abb253(33)
      acc253(28)=abb253(34)
      acc253(29)=abb253(35)
      acc253(30)=abb253(36)
      acc253(31)=abb253(37)
      acc253(32)=abb253(38)
      acc253(33)=abb253(39)
      acc253(34)=abb253(40)
      acc253(35)=abb253(41)
      acc253(36)=abb253(43)
      acc253(37)=abb253(44)
      acc253(38)=abb253(45)
      acc253(39)=abb253(46)
      acc253(40)=abb253(47)
      acc253(41)=abb253(48)
      acc253(42)=abb253(49)
      acc253(43)=abb253(50)
      acc253(44)=abb253(51)
      acc253(45)=abb253(52)
      acc253(46)=abb253(53)
      acc253(47)=abb253(54)
      acc253(48)=abb253(55)
      acc253(49)=abb253(56)
      acc253(50)=abb253(57)
      acc253(51)=abb253(58)
      acc253(52)=abb253(59)
      acc253(53)=abb253(60)
      acc253(54)=Qspvae2k1*acc253(2)
      acc253(55)=Qspvae2l3*acc253(49)
      acc253(54)=acc253(55)+acc253(11)+acc253(54)
      acc253(54)=Qspval4e1*acc253(54)
      acc253(55)=Qspvae2k1*acc253(26)
      acc253(56)=Qspvae2l3*acc253(42)
      acc253(55)=acc253(56)+acc253(48)+acc253(55)
      acc253(55)=Qspval5e1*acc253(55)
      acc253(56)=Qspval4e1*acc253(1)
      acc253(57)=Qspval5e1*acc253(32)
      acc253(56)=acc253(57)+acc253(12)+acc253(56)
      acc253(56)=Qspvae2k2*acc253(56)
      acc253(57)=Qspval4k2*acc253(28)
      acc253(58)=Qspval5k2*acc253(5)
      acc253(59)=Qspvak1k2*acc253(21)
      acc253(60)=Qspvak1e1*acc253(46)
      acc253(61)=-Qspval3e1*acc253(43)
      acc253(62)=Qspvae2l3*acc253(35)
      acc253(63)=QspQ*acc253(34)
      acc253(54)=acc253(56)+acc253(63)+acc253(55)+acc253(54)+acc253(62)+acc253(&
      &61)+acc253(60)+acc253(59)+acc253(58)+acc253(22)+acc253(57)
      acc253(54)=Qspvae1e2*acc253(54)
      acc253(55)=Qspvae1k2*acc253(41)
      acc253(56)=QspQ*acc253(37)
      acc253(55)=acc253(56)+acc253(3)+acc253(55)
      acc253(55)=Qspval5e2*acc253(55)
      acc253(56)=-Qspval4k1*acc253(44)
      acc253(57)=Qspvae1k1*acc253(38)
      acc253(58)=Qspvae1l3*acc253(53)
      acc253(59)=Qspvae1k2*acc253(40)
      acc253(60)=Qspval4e2*acc253(25)
      acc253(61)=Qspval4e2*acc253(52)
      acc253(61)=acc253(31)+acc253(61)
      acc253(61)=QspQ*acc253(61)
      acc253(55)=acc253(55)+acc253(61)+acc253(60)+acc253(59)+acc253(58)+acc253(&
      &57)+acc253(7)+acc253(56)
      acc253(55)=Qspvae2e1*acc253(55)
      acc253(56)=-Qspval4k1*acc253(36)
      acc253(57)=Qspvae1k1*acc253(30)
      acc253(58)=Qspvae1l3*acc253(13)
      acc253(59)=Qspvae1k2*acc253(33)
      acc253(60)=QspQ*acc253(18)
      acc253(56)=acc253(60)+acc253(59)+acc253(58)+acc253(57)+acc253(14)+acc253(&
      &56)
      acc253(56)=Qspvae2k2*acc253(56)
      acc253(57)=Qspvak1k2*acc253(8)
      acc253(58)=Qspvak1e1*acc253(47)
      acc253(59)=-Qspval3e1*acc253(17)
      acc253(60)=QspQ*acc253(39)
      acc253(57)=acc253(60)+acc253(59)+acc253(58)+acc253(9)+acc253(57)
      acc253(57)=Qspval5e2*acc253(57)
      acc253(58)=Qspval4k2*acc253(27)
      acc253(59)=Qspval5k2*acc253(19)
      acc253(60)=Qspvak1k2*acc253(29)
      acc253(61)=Qspvak1e1*acc253(45)
      acc253(62)=Qspvae2k1*acc253(16)
      acc253(63)=Qspval3e1*acc253(10)
      acc253(64)=-Qspvae1k2*acc253(23)
      acc253(65)=Qspvae2l3*acc253(15)
      acc253(66)=Qspval4e2*acc253(50)
      acc253(67)=Qspval4e1*acc253(4)
      acc253(68)=Qspval5e1*acc253(24)
      acc253(69)=Qspval4e2*acc253(51)
      acc253(69)=acc253(20)+acc253(69)
      acc253(69)=QspQ*acc253(69)
      brack=acc253(6)+acc253(54)+acc253(55)+acc253(56)+acc253(57)+acc253(58)+ac&
      &c253(59)+acc253(60)+acc253(61)+acc253(62)+acc253(63)+acc253(64)+acc253(6&
      &5)+acc253(66)+acc253(67)+acc253(68)+acc253(69)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d253h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd253h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d253
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d253 = 0.0_ki
      d253 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d253, ki), aimag(d253), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d253h0l1
