module     p2_gg_httbar_d261h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d261h12l1.f90
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
      use p2_gg_httbar_abbrevd261h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc261(68)
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: QspQ
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l4
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      QspQ = dotproduct(Q,Q)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      acc261(1)=abb261(7)
      acc261(2)=abb261(8)
      acc261(3)=abb261(9)
      acc261(4)=abb261(10)
      acc261(5)=abb261(11)
      acc261(6)=abb261(12)
      acc261(7)=abb261(13)
      acc261(8)=abb261(14)
      acc261(9)=abb261(15)
      acc261(10)=abb261(16)
      acc261(11)=abb261(17)
      acc261(12)=abb261(18)
      acc261(13)=abb261(19)
      acc261(14)=abb261(20)
      acc261(15)=abb261(21)
      acc261(16)=abb261(22)
      acc261(17)=abb261(23)
      acc261(18)=abb261(24)
      acc261(19)=abb261(25)
      acc261(20)=abb261(26)
      acc261(21)=abb261(27)
      acc261(22)=abb261(28)
      acc261(23)=abb261(29)
      acc261(24)=abb261(30)
      acc261(25)=abb261(31)
      acc261(26)=abb261(32)
      acc261(27)=abb261(33)
      acc261(28)=abb261(34)
      acc261(29)=abb261(35)
      acc261(30)=abb261(37)
      acc261(31)=abb261(38)
      acc261(32)=abb261(39)
      acc261(33)=abb261(40)
      acc261(34)=abb261(41)
      acc261(35)=abb261(43)
      acc261(36)=abb261(44)
      acc261(37)=abb261(45)
      acc261(38)=abb261(46)
      acc261(39)=abb261(47)
      acc261(40)=abb261(48)
      acc261(41)=abb261(49)
      acc261(42)=abb261(51)
      acc261(43)=abb261(52)
      acc261(44)=abb261(53)
      acc261(45)=abb261(54)
      acc261(46)=abb261(55)
      acc261(47)=abb261(56)
      acc261(48)=abb261(58)
      acc261(49)=abb261(59)
      acc261(50)=abb261(60)
      acc261(51)=abb261(62)
      acc261(52)=abb261(65)
      acc261(53)=Qspvak2l4*acc261(13)
      acc261(54)=Qspvak2l5*acc261(15)
      acc261(55)=Qspval3l4*acc261(48)
      acc261(56)=Qspval3l5*acc261(31)
      acc261(57)=Qspvak1e2*acc261(39)
      acc261(58)=Qspvae1l4*acc261(28)
      acc261(59)=Qspvae2l5*acc261(9)
      acc261(60)=Qspvak2e1*acc261(18)
      acc261(61)=Qspvae1k2*acc261(5)
      acc261(62)=Qspvak2e2*acc261(49)
      acc261(63)=Qspvae1l3*acc261(40)
      acc261(64)=QspQ*acc261(27)
      acc261(53)=acc261(64)+acc261(63)+acc261(62)+acc261(61)+acc261(60)+acc261(&
      &59)+acc261(58)+acc261(57)+acc261(56)+acc261(55)+acc261(54)+acc261(23)+ac&
      &c261(53)
      acc261(53)=QspQ*acc261(53)
      acc261(54)=Qspvak2l4*acc261(26)
      acc261(55)=-Qspvak2l5*acc261(38)
      acc261(56)=Qspval3l4*acc261(11)
      acc261(57)=Qspval3l5*acc261(44)
      acc261(58)=Qspvak1e2*acc261(43)
      acc261(59)=Qspvak2e1*acc261(10)
      acc261(60)=Qspvak2e2*acc261(50)
      acc261(54)=acc261(60)+acc261(59)+acc261(58)+acc261(57)+acc261(56)+acc261(&
      &55)+acc261(21)+acc261(54)
      acc261(54)=Qspvae1l3*acc261(54)
      acc261(55)=Qspval3l4*acc261(1)
      acc261(56)=Qspval3l5*acc261(35)
      acc261(57)=Qspvak1e2*acc261(41)
      acc261(58)=Qspvak2e1*acc261(4)
      acc261(55)=acc261(58)+acc261(57)+acc261(56)+acc261(2)+acc261(55)
      acc261(55)=Qspvae1k2*acc261(55)
      acc261(56)=Qspvae2e1*acc261(25)
      acc261(57)=Qspvae2e1*acc261(52)
      acc261(57)=acc261(51)+acc261(57)
      acc261(57)=Qspvae1l4*acc261(57)
      acc261(58)=Qspvae1k2*acc261(8)
      acc261(56)=acc261(58)+acc261(57)+acc261(42)+acc261(56)
      acc261(56)=Qspvak2e2*acc261(56)
      acc261(57)=Qspval3e1*acc261(45)
      acc261(58)=Qspvae2e1*acc261(30)
      acc261(57)=acc261(58)+acc261(24)+acc261(57)
      acc261(57)=Qspvae1l4*acc261(57)
      acc261(58)=Qspval3e1*acc261(46)
      acc261(59)=Qspvae1e2*acc261(20)
      acc261(58)=acc261(59)+acc261(3)+acc261(58)
      acc261(58)=Qspvae2l5*acc261(58)
      acc261(59)=Qspvae1e2*acc261(16)
      acc261(60)=Qspvae1e2*acc261(34)
      acc261(60)=acc261(32)+acc261(60)
      acc261(60)=Qspvae2l5*acc261(60)
      acc261(59)=acc261(60)+acc261(19)+acc261(59)
      acc261(59)=Qspvak2e1*acc261(59)
      acc261(60)=Qspvae2l4*acc261(6)
      acc261(61)=Qspvak2l4*acc261(12)
      acc261(62)=Qspvak2l5*acc261(14)
      acc261(63)=Qspval3e1*acc261(29)
      acc261(64)=Qspval3l4*acc261(47)
      acc261(65)=Qspval3l5*acc261(22)
      acc261(66)=Qspvak1e2*acc261(37)
      acc261(67)=Qspvae2e1*acc261(33)
      acc261(68)=Qspvae2l4*acc261(17)
      acc261(68)=acc261(36)+acc261(68)
      acc261(68)=Qspvae1e2*acc261(68)
      brack=acc261(7)+acc261(53)+acc261(54)+acc261(55)+acc261(56)+acc261(57)+ac&
      &c261(58)+acc261(59)+acc261(60)+acc261(61)+acc261(62)+acc261(63)+acc261(6&
      &4)+acc261(65)+acc261(66)+acc261(67)+acc261(68)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d261h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd261h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d261
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d261 = 0.0_ki
      d261 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d261, ki), aimag(d261), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d261h12l1
