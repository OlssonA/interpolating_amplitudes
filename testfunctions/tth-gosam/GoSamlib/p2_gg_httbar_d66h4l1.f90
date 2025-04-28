module     p2_gg_httbar_d66h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d66h4l1.f90
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
      use p2_gg_httbar_abbrevd66h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc66(59)
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3k2
      complex(ki) :: QspQ
      complex(ki) :: Qspk1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspl3
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval5k2
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      QspQ = dotproduct(Q,Q)
      Qspk1 = dotproduct(Q,k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspl3 = dotproduct(Q,l3)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      acc66(1)=abb66(9)
      acc66(2)=abb66(10)
      acc66(3)=abb66(11)
      acc66(4)=abb66(12)
      acc66(5)=abb66(13)
      acc66(6)=abb66(14)
      acc66(7)=abb66(15)
      acc66(8)=abb66(16)
      acc66(9)=abb66(17)
      acc66(10)=abb66(18)
      acc66(11)=abb66(19)
      acc66(12)=abb66(20)
      acc66(13)=abb66(21)
      acc66(14)=abb66(22)
      acc66(15)=abb66(23)
      acc66(16)=abb66(24)
      acc66(17)=abb66(25)
      acc66(18)=abb66(26)
      acc66(19)=abb66(28)
      acc66(20)=abb66(29)
      acc66(21)=abb66(30)
      acc66(22)=abb66(31)
      acc66(23)=abb66(32)
      acc66(24)=abb66(33)
      acc66(25)=abb66(34)
      acc66(26)=abb66(35)
      acc66(27)=abb66(38)
      acc66(28)=abb66(39)
      acc66(29)=abb66(40)
      acc66(30)=abb66(41)
      acc66(31)=abb66(42)
      acc66(32)=abb66(43)
      acc66(33)=abb66(48)
      acc66(34)=abb66(50)
      acc66(35)=abb66(51)
      acc66(36)=abb66(52)
      acc66(37)=abb66(56)
      acc66(38)=abb66(58)
      acc66(39)=Qspval3l4*acc66(37)
      acc66(40)=Qspval5l4*acc66(24)
      acc66(41)=Qspvak2l3*acc66(14)
      acc66(42)=Qspval5l3*acc66(15)
      acc66(43)=Qspval3k2*acc66(30)
      acc66(39)=-acc66(43)-acc66(39)+acc66(40)+acc66(41)+acc66(42)
      acc66(40)=QspQ*acc66(21)
      acc66(40)=acc66(40)+acc66(35)+acc66(39)
      acc66(40)=Qspk1*acc66(40)
      acc66(41)=Qspvak1k2*acc66(16)
      acc66(42)=QspQ*acc66(11)
      acc66(43)=Qspk1*acc66(26)
      acc66(44)=Qspk2*acc66(12)
      acc66(39)=acc66(44)+acc66(43)+acc66(42)+acc66(1)+acc66(41)-acc66(39)
      acc66(39)=Qspk2*acc66(39)
      acc66(41)=-Qspvak1l4*acc66(31)
      acc66(42)=acc66(38)*Qspval5k1
      acc66(43)=Qspvak1k2*acc66(27)
      acc66(44)=Qspvak2k1*acc66(22)
      acc66(41)=acc66(44)+acc66(43)+acc66(42)+acc66(10)+acc66(41)
      acc66(41)=QspQ*acc66(41)
      acc66(42)=acc66(27)*Qspvak1l3
      acc66(43)=Qspvak2l3*acc66(18)
      acc66(42)=acc66(43)+acc66(3)+acc66(42)
      acc66(42)=Qspval3k2*acc66(42)
      acc66(43)=acc66(36)*Qspval4k1
      acc66(44)=acc66(33)*Qspl3
      acc66(45)=acc66(28)*Qspvak1l5
      acc66(46)=acc66(8)*Qspval3l5
      acc66(47)=acc66(7)*Qspvak2l5
      acc66(48)=acc66(5)*Qspvak2l4
      acc66(49)=Qspvak1l3*acc66(13)
      acc66(50)=Qspvak1l4*acc66(29)
      acc66(51)=Qspval3k1*acc66(17)
      acc66(52)=Qspval5k1*acc66(34)
      acc66(53)=Qspval5k2*acc66(32)
      acc66(54)=Qspvak1k2*acc66(6)
      acc66(55)=Qspval5k2*acc66(20)
      acc66(55)=acc66(19)+acc66(55)
      acc66(55)=Qspvak2k1*acc66(55)
      acc66(56)=Qspval3l4*acc66(9)
      acc66(57)=Qspval5l4*acc66(23)
      acc66(58)=Qspvak2l3*acc66(2)
      acc66(59)=acc66(38)*Qspval3k1
      acc66(59)=acc66(25)+acc66(59)
      acc66(59)=Qspval5l3*acc66(59)
      brack=acc66(4)+acc66(39)+acc66(40)+acc66(41)+acc66(42)+acc66(43)+acc66(44&
      &)+acc66(45)+acc66(46)+acc66(47)+acc66(48)+acc66(49)+acc66(50)+acc66(51)+&
      &acc66(52)+acc66(53)+acc66(54)+acc66(55)+acc66(56)+acc66(57)+acc66(58)+ac&
      &c66(59)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d66h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd66h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d66
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d66 = 0.0_ki
      d66 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d66, ki), aimag(d66), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d66h4l1
