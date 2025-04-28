module     p2_gg_httbar_d13h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d13h8l1.f90
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
      use p2_gg_httbar_abbrevd13h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc13(41)
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspk1
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl4
      complex(ki) :: QspQ
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l5
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspk1 = dotproduct(Q,k1)
      Qspk2 = dotproduct(Q,k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl4 = dotproduct(Q,l4)
      QspQ = dotproduct(Q,Q)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      acc13(1)=abb13(9)
      acc13(2)=abb13(10)
      acc13(3)=abb13(11)
      acc13(4)=abb13(12)
      acc13(5)=abb13(13)
      acc13(6)=abb13(14)
      acc13(7)=abb13(15)
      acc13(8)=abb13(16)
      acc13(9)=abb13(17)
      acc13(10)=abb13(18)
      acc13(11)=abb13(19)
      acc13(12)=abb13(20)
      acc13(13)=abb13(21)
      acc13(14)=abb13(22)
      acc13(15)=abb13(23)
      acc13(16)=abb13(24)
      acc13(17)=abb13(25)
      acc13(18)=abb13(27)
      acc13(19)=abb13(29)
      acc13(20)=abb13(31)
      acc13(21)=abb13(32)
      acc13(22)=abb13(33)
      acc13(23)=abb13(34)
      acc13(24)=Qspval3k2*acc13(21)
      acc13(25)=Qspval4l3*acc13(11)
      acc13(26)=Qspval4l5*acc13(8)
      acc13(24)=acc13(26)+acc13(24)-acc13(25)
      acc13(25)=acc13(3)-acc13(24)
      acc13(25)=Qspk1*acc13(25)
      acc13(26)=Qspk2-Qspk1
      acc13(26)=acc13(23)*acc13(26)
      acc13(24)=acc13(1)+acc13(26)+acc13(24)
      acc13(24)=Qspk2*acc13(24)
      acc13(26)=acc13(22)*Qspval3l4
      acc13(27)=acc13(20)*Qspvak2l4
      acc13(28)=acc13(18)*Qspvak1l5
      acc13(29)=acc13(17)*Qspval4k1
      acc13(30)=acc13(16)*Qspvak1k2
      acc13(31)=acc13(15)*Qspvak2l3
      acc13(32)=acc13(14)*Qspval4k2
      acc13(33)=acc13(10)*Qspvak1l3
      acc13(34)=acc13(9)*Qspl4
      acc13(35)=acc13(7)*QspQ
      acc13(36)=acc13(5)*Qspval3k1
      acc13(37)=acc13(4)*Qspvak2k1
      acc13(38)=acc13(2)*Qspvak2l5
      acc13(39)=Qspval3k2*acc13(19)
      acc13(40)=Qspval4l3*acc13(13)
      acc13(41)=Qspval4l5*acc13(12)
      brack=acc13(6)+acc13(24)+acc13(25)+acc13(26)+acc13(27)+acc13(28)+acc13(29&
      &)+acc13(30)+acc13(31)+acc13(32)+acc13(33)+acc13(34)+acc13(35)+acc13(36)+&
      &acc13(37)+acc13(38)+acc13(39)+acc13(40)+acc13(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d13h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd13h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d13
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d13 = 0.0_ki
      d13 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d13, ki), aimag(d13), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d13h8l1
