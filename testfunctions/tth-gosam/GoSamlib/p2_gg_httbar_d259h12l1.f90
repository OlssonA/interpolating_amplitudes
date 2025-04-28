module     p2_gg_httbar_d259h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d259h12l1.f90
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
      use p2_gg_httbar_abbrevd259h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc259(58)
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: QspQ
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      QspQ = dotproduct(Q,Q)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      acc259(1)=abb259(7)
      acc259(2)=abb259(8)
      acc259(3)=abb259(9)
      acc259(4)=abb259(10)
      acc259(5)=abb259(11)
      acc259(6)=abb259(12)
      acc259(7)=abb259(13)
      acc259(8)=abb259(14)
      acc259(9)=abb259(15)
      acc259(10)=abb259(16)
      acc259(11)=abb259(17)
      acc259(12)=abb259(18)
      acc259(13)=abb259(19)
      acc259(14)=abb259(20)
      acc259(15)=abb259(21)
      acc259(16)=abb259(22)
      acc259(17)=abb259(23)
      acc259(18)=abb259(24)
      acc259(19)=abb259(25)
      acc259(20)=abb259(26)
      acc259(21)=abb259(27)
      acc259(22)=abb259(29)
      acc259(23)=abb259(30)
      acc259(24)=abb259(31)
      acc259(25)=abb259(32)
      acc259(26)=abb259(33)
      acc259(27)=abb259(34)
      acc259(28)=abb259(35)
      acc259(29)=abb259(36)
      acc259(30)=abb259(37)
      acc259(31)=abb259(38)
      acc259(32)=abb259(39)
      acc259(33)=abb259(40)
      acc259(34)=abb259(41)
      acc259(35)=abb259(42)
      acc259(36)=abb259(45)
      acc259(37)=abb259(46)
      acc259(38)=abb259(48)
      acc259(39)=abb259(51)
      acc259(40)=abb259(53)
      acc259(41)=abb259(54)
      acc259(42)=abb259(57)
      acc259(43)=abb259(58)
      acc259(44)=abb259(61)
      acc259(45)=abb259(62)
      acc259(46)=Qspvak1l4*acc259(34)
      acc259(47)=Qspvak1l5*acc259(25)
      acc259(48)=Qspval3e1*acc259(30)
      acc259(49)=Qspvae2l4*acc259(9)
      acc259(50)=Qspvae1l5*acc259(26)
      acc259(51)=Qspvak2e1*acc259(12)
      acc259(52)=Qspvae1k2*acc259(33)
      acc259(53)=Qspvak2e2*acc259(40)
      acc259(54)=Qspvae1l3*acc259(39)
      acc259(55)=QspQ*acc259(41)
      acc259(46)=acc259(55)+acc259(54)+acc259(53)+acc259(52)+acc259(51)+acc259(&
      &50)+acc259(49)+acc259(48)+acc259(47)+acc259(24)+acc259(46)
      acc259(46)=QspQ*acc259(46)
      acc259(47)=Qspvak1l4*acc259(35)
      acc259(48)=Qspvak1l5*acc259(32)
      acc259(49)=Qspval3e1*acc259(20)
      acc259(50)=Qspvak2e1*acc259(11)
      acc259(51)=Qspvak2e2*acc259(42)
      acc259(47)=acc259(51)+acc259(50)+acc259(49)+acc259(48)+acc259(15)+acc259(&
      &47)
      acc259(47)=Qspvae1l3*acc259(47)
      acc259(48)=Qspvak1l4*acc259(17)
      acc259(49)=Qspvak1l5*acc259(6)
      acc259(50)=Qspval3e1*acc259(21)
      acc259(51)=Qspvak2e1*acc259(1)
      acc259(48)=acc259(51)+acc259(50)+acc259(49)+acc259(23)+acc259(48)
      acc259(48)=Qspvae1k2*acc259(48)
      acc259(49)=Qspvae2e1*acc259(44)
      acc259(50)=Qspvae2e1*acc259(45)
      acc259(50)=acc259(38)+acc259(50)
      acc259(50)=Qspvae1l5*acc259(50)
      acc259(51)=Qspvae1k2*acc259(27)
      acc259(49)=acc259(51)+acc259(50)+acc259(2)+acc259(49)
      acc259(49)=Qspvak2e2*acc259(49)
      acc259(50)=Qspvae1e2*acc259(36)
      acc259(51)=Qspval3e1*acc259(43)
      acc259(50)=acc259(51)+acc259(4)+acc259(50)
      acc259(50)=Qspvae2l4*acc259(50)
      acc259(51)=Qspvae2e1*acc259(31)
      acc259(52)=Qspval3e1*acc259(28)
      acc259(51)=acc259(52)+acc259(29)+acc259(51)
      acc259(51)=Qspvae1l5*acc259(51)
      acc259(52)=Qspvae1e2*acc259(13)
      acc259(53)=Qspvae1e2*acc259(3)
      acc259(53)=acc259(14)+acc259(53)
      acc259(53)=Qspvae2l4*acc259(53)
      acc259(52)=acc259(53)+acc259(10)+acc259(52)
      acc259(52)=Qspvak2e1*acc259(52)
      acc259(53)=Qspvae1l4*acc259(37)
      acc259(54)=Qspvak1l4*acc259(7)
      acc259(55)=Qspvak1l5*acc259(8)
      acc259(56)=Qspvae1e2*acc259(19)
      acc259(57)=Qspvae1l4*acc259(18)
      acc259(57)=acc259(16)+acc259(57)
      acc259(57)=Qspvae2e1*acc259(57)
      acc259(58)=Qspval3e1*acc259(22)
      brack=acc259(5)+acc259(46)+acc259(47)+acc259(48)+acc259(49)+acc259(50)+ac&
      &c259(51)+acc259(52)+acc259(53)+acc259(54)+acc259(55)+acc259(56)+acc259(5&
      &7)+acc259(58)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d259h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd259h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d259
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d259 = 0.0_ki
      d259 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d259, ki), aimag(d259), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d259h12l1
