module     p2_gg_httbar_d13h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d13h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd13h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc13(41)
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspk1
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl4
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak2l4
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspk1 = dotproduct(Q,k1)
      Qspk2 = dotproduct(Q,k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl4 = dotproduct(Q,l4)
      QspQ = dotproduct(Q,Q)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
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
      acc13(16)=abb13(25)
      acc13(17)=abb13(26)
      acc13(18)=abb13(30)
      acc13(19)=abb13(32)
      acc13(20)=abb13(34)
      acc13(21)=abb13(36)
      acc13(22)=abb13(38)
      acc13(23)=abb13(39)
      acc13(24)=Qspvak2l3*acc13(15)
      acc13(25)=Qspval3l4*acc13(11)
      acc13(26)=Qspval5l4*acc13(8)
      acc13(24)=acc13(26)+acc13(24)-acc13(25)
      acc13(25)=acc13(3)-acc13(24)
      acc13(25)=Qspk1*acc13(25)
      acc13(26)=Qspk2-Qspk1
      acc13(26)=acc13(20)*acc13(26)
      acc13(24)=acc13(1)+acc13(26)+acc13(24)
      acc13(24)=Qspk2*acc13(24)
      acc13(26)=acc13(22)*Qspval4k2
      acc13(27)=acc13(21)*Qspval4l3
      acc13(28)=acc13(19)*Qspval5k2
      acc13(29)=acc13(18)*Qspval5k1
      acc13(30)=acc13(17)*Qspvak2k1
      acc13(31)=acc13(14)*Qspval3k2
      acc13(32)=acc13(13)*Qspval3k1
      acc13(33)=acc13(10)*Qspvak1l3
      acc13(34)=acc13(9)*Qspl4
      acc13(35)=acc13(7)*QspQ
      acc13(36)=acc13(5)*Qspvak1l4
      acc13(37)=acc13(4)*Qspvak1k2
      acc13(38)=acc13(2)*Qspvak2l4
      acc13(39)=Qspvak2l3*acc13(16)
      acc13(40)=Qspval3l4*acc13(23)
      acc13(41)=Qspval5l4*acc13(12)
      brack=acc13(6)+acc13(24)+acc13(25)+acc13(26)+acc13(27)+acc13(28)+acc13(29&
      &)+acc13(30)+acc13(31)+acc13(32)+acc13(33)+acc13(34)+acc13(35)+acc13(36)+&
      &acc13(37)+acc13(38)+acc13(39)+acc13(40)+acc13(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d13h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd13h4_qp
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
end module p2_gg_httbar_d13h4l1_qp
